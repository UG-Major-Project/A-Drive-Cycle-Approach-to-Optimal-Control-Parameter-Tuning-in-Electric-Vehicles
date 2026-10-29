"""
Forward-facing EV model (Python port of the Simulink model)
Blocks: DriverPI -> ECU pedal/brake -> Motor & Power -> Vehicle Dynamics -> Battery
All blocks use dt = 0.1 s and carry their own state, like persistent variables
in the MATLAB Function blocks. v_act is the one-step-delayed vehicle speed
(the Unit Delay block that breaks the algebraic loop).

Usage:
    python ev_forward_model.py                 # synthetic cycle
    python ev_forward_model.py cycle.csv       # CSV with columns: time_s, speed_mps
"""
import sys
import numpy as np
_trapz = getattr(np, 'trapezoid', None) or np.trapz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ecu_optimizer import ThrottleRateLimiter, TorqueMap, PowerRateLimiter


DT = 0.1


# ---------------------------------------------------------------- Driver PI
class DriverPI:
    def __init__(self, Kp=0.5, Ki=0.1, dt=DT):
        self.Kp, self.Ki, self.dt = Kp, Ki, dt
        self.I = 0.0

    def step(self, v_ref, v):
        e = v_ref - v
        u_unsat = self.Kp * e + self.I
        u = max(min(u_unsat, 1.0), -1.0)
        # anti-windup: do not integrate while saturated and pushing further
        if not (u != u_unsat and np.sign(e) == np.sign(u_unsat)):
            self.I += self.Ki * e * self.dt
        # standstill: clear integrator so the car doesn't creep
        if v_ref < 0.1 and v < 0.1:
            self.I = 0.0
        return u


# ---------------------------------------------------------- ECU pedal/brake
class ECUPedalBrake:
    """
    The ThrottleRateLimiter (if given) is the ONLY rate limiter here. The old
    built-in rate_max limiter was removed, so the "without ECU" baseline
    now has no pedal rate limiting at all.
    """
    def __init__(self, pedal_exp=1.0, F_brake_max=6000.0,
                 Tpeak=375.0, r_wheel=0.30, G=9.0, dt=DT, limiter=None,
                 torque_map=None, power_limiter=None):
        self.torque_map = torque_map
        self.power_limiter = power_limiter
        self.pedal_exp = pedal_exp
        self.F_brake_max, self.Tpeak = F_brake_max, Tpeak
        self.r, self.G, self.dt = r_wheel, G, dt
        self.limiter = limiter

    def step(self, u, v):
        # ---- ECU hook: rate limiter (your research goes here)
        if self.limiter is not None:
            u = self.limiter.apply_throttle_curve(u, v)

        cmd = np.sign(u) * abs(u) ** self.pedal_exp

        if cmd >= 0:
            T_req = T_raw = cmd * self.Tpeak
            if self.torque_map is not None:
                T_req = min(T_req, self.torque_map.cap(v))
            F_brake_total = 0.0
        else:
            F_brake_total = -cmd * self.F_brake_max
            if v > 15:
                ratio = 1.0
            elif v > 5:
                ratio = (v - 5) / 10.0
            else:
                ratio = 0.0
            F_regen_req = F_brake_total * ratio
            T_req = T_raw = -F_regen_req * self.r / self.G

        # dP/dt limiter acts on SIGNED torque, so tip-in -> regen (and back)
        # ramps through zero instead of jumping. Friction blending in
        # VehicleDynamics still delivers the requested brake force.
        if self.power_limiter is not None:
            T_req = self.power_limiter.apply(T_req, v)

        if cmd >= 0 and T_req != T_raw:
            # report the effective pedal so the PI anti-windup sees the limit
            u = np.sign(T_req) * (abs(T_req) / self.Tpeak) ** (1.0 / self.pedal_exp)
        return T_req, F_brake_total, u


# ------------------------------------------------------------ Motor & Power
class MotorPower:
    def __init__(self, dt=DT):
        self.dt = dt
        self.r_wheel = 0.30
        self.G = 9.0
        self.Kt = 1.2
        self.R_phase = 0.05
        self.tau_elec = 0.05
        self.k_iron = 2e-5
        self.Nbase_rpm = 5000.0
        self.Tpeak = 375.0
        self.C_th = 4000.0
        self.h_cool = 25.0
        self.T_amb = 25.0
        self.T_derate_start = 120.0
        self.T_max = 150.0
        self.regen_eff = 0.85
        self.I_q = 0.0
        self.T_temp = 25.0

    def step(self, T_req, v):
        omega = (v / self.r_wheel) * self.G
        rpm = omega * 60 / (2 * np.pi)

        # speed-dependent torque limit
        if abs(rpm) <= self.Nbase_rpm:
            T_limit = self.Tpeak
        else:
            P_peak = self.Tpeak * (self.Nbase_rpm * np.pi / 30)
            T_limit = P_peak / max(abs(omega), 1e-3)

        # thermal derating
        if self.T_temp > self.T_max:
            derate = 0.0
        elif self.T_temp > self.T_derate_start:
            derate = (self.T_max - self.T_temp) / (self.T_max - self.T_derate_start)
        else:
            derate = 1.0
        T_limit *= derate
        T_cmd = max(min(T_req, T_limit), -T_limit)

        # current dynamics (exact discretisation of 1st-order lag)
        I_cmd = T_cmd / self.Kt
        alpha = 1 - np.exp(-self.dt / self.tau_elec)
        self.I_q += alpha * (I_cmd - self.I_q)
        T_actual = self.Kt * self.I_q

        # losses
        P_copper = self.I_q ** 2 * self.R_phase
        P_iron = self.k_iron * omega ** 2
        P_mech = T_actual * omega

        if P_mech >= 0:
            P_motor = P_mech + P_copper + P_iron
            P_regen = 0.0
        else:
            P_regen = max(-P_mech * self.regen_eff - P_copper - P_iron, 0.0)
            P_motor = 0.0

        # thermal update
        P_loss = P_copper + P_iron
        self.T_temp += (self.dt / self.C_th) * (
            P_loss - self.h_cool * (self.T_temp - self.T_amb))

        F_motor = T_actual * self.G / self.r_wheel
        return dict(P_motor=P_motor, P_regen=P_regen, I_q=self.I_q,
                    T_temp=self.T_temp, T_actual=T_actual, F_motor=F_motor)


# --------------------------------------------------------- Vehicle Dynamics
class VehicleDynamics:
    def __init__(self, dt=DT):
        self.dt = dt
        self.m, self.Cd, self.A = 1500.0, 0.28, 2.2
        self.rho, self.Crr, self.g = 1.225, 0.01, 9.81
        self.v = 0.0
        self.dist = 0.0

    def step(self, F_motor, F_brake_total, grade=0.0):
        F_regen_act = max(-F_motor, 0.0)
        F_fric = max(F_brake_total - F_regen_act, 0.0)

        F_aero = 0.5 * self.rho * self.Cd * self.A * self.v ** 2
        F_roll = self.m * self.g * self.Crr
        F_grade = self.m * self.g * np.sin(grade)

        a = (F_motor - F_fric - F_aero - F_roll - F_grade) / self.m
        v_new = max(self.v + a * self.dt, 0.0)
        if v_new == 0.0 and self.v == 0.0:
            a = 0.0
        self.dist += 0.5 * (self.v + v_new) * self.dt
        self.v = v_new
        return self.v, a, self.dist


# -------------------------------------------------------------- Battery SoC
class Battery:
    def __init__(self, dt=DT):
        self.dt = dt
        self.Q = 50 * 3600.0
        self.R0, self.R1, self.C1 = 0.05, 0.03, 2000.0
        self.SoC_bp = np.array([0, .1, .2, .4, .6, .8, .9, 1.0])
        self.Voc_bp = np.array([320, 345, 355, 362, 367, 372, 378, 392.0])
        self.SoC = 1.0
        self.Vc1 = 0.0

    def _voc(self, soc):
        # linear interpolation with linear extrapolation (matches interp1 'extrap')
        x, y = self.SoC_bp, self.Voc_bp
        if soc <= x[0]:
            return y[0] + (soc - x[0]) * (y[1] - y[0]) / (x[1] - x[0])
        if soc >= x[-1]:
            return y[-1] + (soc - x[-1]) * (y[-1] - y[-2]) / (x[-1] - x[-2])
        return float(np.interp(soc, x, y))

    def step(self, P_motor, P_regen):
        net_power = P_motor - P_regen
        Voc = self._voc(self.SoC)

        a, b, c = -self.R0, (Voc - self.Vc1), -net_power
        disc = max(b * b - 4 * a * c, 0.0)
        I1 = (-b + np.sqrt(disc)) / (2 * a)
        I2 = (-b - np.sqrt(disc)) / (2 * a)
        I_batt = I1 if abs(I1) <= abs(I2) else I2

        self.Vc1 += (self.dt / self.C1) * (I_batt - self.Vc1 / self.R1)
        V_terminal = Voc - I_batt * self.R0 - self.Vc1
        self.SoC = min(max(self.SoC - I_batt * self.dt / self.Q, 0.0), 1.0)
        return dict(I_batt=I_batt, net_power=net_power,
                    V_terminal=V_terminal, SoC=self.SoC)


def make_battery():
    """Battery factory. BATTERY_MODEL (set in the config block at the bottom of
    this file, or by a sweep's --battery flag): "spm" = SPPy single particle
    model, "ecm" = the original Thevenin model above."""
    if BATTERY_MODEL == "ecm":
        return Battery()
    from battery_spm import SPMBattery
    return SPMBattery(dt=DT, **BATTERY_KW)


# -------------------------------------------------------------- Drive cycle
def synthetic_cycle(dt=DT):
    """Simple urban + highway style test cycle (time_s, speed_mps)."""
    kmh = [(0, 0), (10, 0), (25, 40), (40, 40), (55, 0), (65, 0),
           (85, 60), (110, 60), (125, 100), (160, 100), (185, 50),
           (200, 50), (225, 0), (240, 0)]
    t_pts = [p[0] for p in kmh]
    v_pts = [p[1] / 3.6 for p in kmh]
    t = np.arange(0, t_pts[-1] + dt, dt)
    return t, np.interp(t, t_pts, v_pts)


def load_cycle(path, dt=DT):
    data = np.loadtxt(path, skiprows=1)
    t_raw, v_raw = data[:, 0], data[:, 1]          # v in m/s, check units!
    v_raw *= 0.44705 # conversion to m/s
    t = np.arange(t_raw[0], t_raw[-1] + dt, dt)
    return t, np.interp(t, t_raw, v_raw)


# ---------------------------------------------------------------- Simulation
def simulate(t, v_ref, is_ecu, driver=None, open_loop_u=None,
             limiter_kwargs=None, Kb_aw=1.0, torque_map=None, use_limiter=True,
             power_limiter_kwargs=None, battery_factory=None):
    """
    Closed loop (Method 1): leave open_loop_u=None.
    Open loop (Method 2): pass an array u(t) in [-1, 1]; the PI is bypassed.

    limiter_kwargs: dict passed to ThrottleRateLimiter (rise_per_s, fall_per_s,
                    v_knee, v_blend, hf_scale). Only used when is_ecu=True.
    torque_map:     optional TorqueMap (speed-indexed torque ceiling), ECU only.
    use_limiter:    set False to run the ECU with the TorqueMap but no rate limiter.
    power_limiter_kwargs: dict for PowerRateLimiter (dpdt_up, fall_ratio, ...);
                    None disables it. ECU only.
    battery_factory: callable returning a battery object; default make_battery().
    Kb_aw:          back-calculation gain feeding (u_ecu - u_pi) into the PI
                    integrator so it does not wind up against the limiter.
    """
    driver = driver or DriverPI()
    limiter = None
    if is_ecu and use_limiter:
        limiter = ThrottleRateLimiter(dt=DT, **(limiter_kwargs or {}))
    plim = None
    if is_ecu and power_limiter_kwargs is not None:
        plim = PowerRateLimiter(dt=DT, **power_limiter_kwargs)
    ecu = ECUPedalBrake(limiter=limiter,
                        torque_map=torque_map if is_ecu else None,
                        power_limiter=plim)
    motor, veh = MotorPower(), VehicleDynamics()
    batt = (battery_factory or make_battery)()

    keys = ["v_ref", "v", "a", "dist", "u", "T_req", "T_actual", "F_motor",
            "F_brake", "P_motor", "P_regen", "I_q", "T_temp", "I_batt",
            "V_terminal", "SoC", "u_new", "v_diff"]
    log = {k: np.zeros(len(t)) for k in keys}

    v_act = 0.0                                     # Unit Delay output
    for k in range(len(t)):
        u = driver.step(v_ref[k], v_act) if open_loop_u is None else open_loop_u[k]
        T_req, F_brake, u_new = ecu.step(u, v_act)
        if open_loop_u is None:
            # back-calculation anti-windup (zero when the limiter is inactive)
            driver.I += Kb_aw * (u_new - u) * DT
        m = motor.step(T_req, v_act)
        v, a, dist = veh.step(m["F_motor"], F_brake)
        v_diff = v_ref[max(k - 1, 0)] - v
        b = batt.step(m["P_motor"], m["P_regen"])

        for key, val in dict(v_ref=v_ref[k], v=v, a=a, dist=dist, u=u,
                             T_req=T_req, T_actual=m["T_actual"],
                             F_motor=m["F_motor"], F_brake=F_brake,
                             P_motor=m["P_motor"], P_regen=m["P_regen"],
                             I_q=m["I_q"], T_temp=m["T_temp"],
                             I_batt=b["I_batt"], V_terminal=b["V_terminal"],
                             SoC=b["SoC"], u_new=u_new, v_diff=v_diff).items():
            log[key][k] = val
        v_act = v                                    # unit delay
    return log


def summarize(t, log):
    err = log["v_ref"] - log["v"]
    rmse = np.sqrt(np.mean(err ** 2))
    d_ref = _trapz(log["v_ref"], t)
    d_act = log["dist"][-1]
    E_kWh = _trapz(log["P_motor"] - log["P_regen"], t) / 3.6e6
    print(f"Speed RMSE      : {rmse:.3f} m/s   (max |err| {np.max(abs(err)):.2f})")
    print(f"Distance        : {d_act:.0f} m actual vs {d_ref:.0f} m reference")
    print(f"Net battery E   : {E_kWh:.3f} kWh   ({E_kWh / max(d_act/1000, 1e-6) * 1000:.0f} Wh/km)")
    print(f"Final SoC       : {log['SoC'][-1]:.4f}")
    print(f"Peak motor temp : {log['T_temp'].max():.2f} degC")


def metrics(t, log):
    """Scalar metrics for judging the limiter (current smoothness vs tracking)."""
    dt = t[1] - t[0]
    I = log["I_batt"]
    err = log["v_ref"] - log["v"]
    d_km = max(log["dist"][-1] / 1000.0, 1e-6)
    E_kWh = _trapz(log["P_motor"] - log["P_regen"], t) / 3.6e6
    return dict(
        peak_I=float(np.max(np.abs(I))),
        rms_I=float(np.sqrt(np.mean(I ** 2))),
        max_dIdt=float(np.max(np.abs(np.diff(I))) / dt),
        rmse=float(np.sqrt(np.mean(err ** 2))),
        Wh_km=float(E_kWh / d_km * 1000.0),
        soc_end=float(log["SoC"][-1]),
    )


def peak_power_rate(t, log, r=0.30, G=9.0):
    """Largest |dP/dt| [W/s] of positive mechanical power in a (baseline) run."""
    P = np.maximum(log["T_actual"], 0.0) * (log["v"] / r * G)
    return float(np.max(np.abs(np.diff(P))) / (t[1] - t[0]))


def calibrate_ecu(t, base_log, torque_scale=0.8, dpdt_frac=0.5,
                  fall_ratio=2.0, bin_w=2.5):
    """
    Derive the ECU settings from a baseline (no-ECU) run, nothing hard-coded:
      - TorqueMap  = per-speed-bin torque demand envelope x torque_scale
      - dP/dt limit = dpdt_frac x the baseline's peak |dP/dt|
    Pass torque_scale=0 / dpdt_frac=0 to disable either one.
    Returns (kwargs for simulate(), reference dP/dt in W/s).
    """
    tm = None
    if torque_scale:
        tm = TorqueMap.from_envelope(base_log["v"], base_log["T_req"],
                                     bin_w=bin_w, scale=torque_scale)
    ref = peak_power_rate(t, base_log)
    pl = dict(dpdt_up=dpdt_frac * ref, fall_ratio=fall_ratio) if dpdt_frac else None
    return dict(use_limiter=False, torque_map=tm, power_limiter_kwargs=pl), ref


def top_didt_events(t, log, n=5):
    """Print the largest battery-current steps and what the drivetrain was doing."""
    dI = np.abs(np.diff(log["I_batt"])) / (t[1] - t[0])
    print("Largest |dI/dt| events:")
    for i in np.argsort(dI)[-n:][::-1] + 1:
        print(f"  t={t[i]:.1f}s  dI/dt={dI[i-1]:.0f} A/s  u={log['u'][i]:.2f}  "
              f"T_req={log['T_req'][i]:.0f} Nm  F_brake={log['F_brake'][i]:.0f} N  "
              f"v={log['v'][i]:.1f} m/s")


def compare(m_ecu, m_base):
    rows = [("Peak |I_batt| [A]", "peak_I", "{:.1f}"),
            ("RMS I_batt [A]", "rms_I", "{:.2f}"),
            ("Max |dI/dt| [A/s]", "max_dIdt", "{:.0f}"),
            ("Speed RMSE [m/s]", "rmse", "{:.3f}"),
            ("Energy [Wh/km]", "Wh_km", "{:.1f}"),
            ("Final SoC", "soc_end", "{:.4f}")]
    print(f"{'':22s}{'With ECU':>12s}{'Without':>12s}{'Change':>10s}")
    for label, key, fmt in rows:
        a, b = m_ecu[key], m_base[key]
        pct = (a - b) / abs(b) * 100 if abs(b) > 1e-12 else float("nan")
        print(f"{label:22s}{fmt.format(a):>12s}{fmt.format(b):>12s}{pct:>+9.1f}%")


def plot(t, log, log1, fname):
    fig, ax = plt.subplots(7, 1, figsize=(10, 13), sharex=True)
    ax[0].plot(t, log["v_ref"] * 3.6, "--", label="v_ref")
    ax[0].plot(t, log["v"] * 3.6, label="v_actual")
    ax[0].set_ylabel("m/s"); ax[0].legend(); ax[0].grid(True)
    ax[1].plot(t, log["u"], label="PI output u")
    ax[1].plot(t, log["u_new"], label="ECU out u")
    ax[1].set_ylabel("pedal (-1..1)"); ax[1].grid(True)
    ax[2].plot(t, log["T_req"], "--", label="T_req")
    ax[2].plot(t, log["T_actual"], label="T_actual")
    ax[2].set_ylabel("Nm"); ax[2].legend(); ax[2].grid(True)
    ax[3].plot(t, (log["P_motor"] - log["P_regen"]) / 1e3)
    ax[3].set_ylabel("battery power kW"); ax[3].grid(True)
    ax[4].plot(t, log["SoC"] * 100)
    ax[4].set_ylabel("SoC %"); ax[4].set_xlabel("time (s)"); ax[4].grid(True)
    ax[5].plot(t, log["v_diff"]);ax[5].set_xlabel("time (s)"); ax[5].grid(True)
    ax[5].set_ylabel("m/s")
    ax[6].plot(t, log["I_batt"], label="With ECU");ax[6].set_xlabel("time (s)"); ax[6].grid(True)
    ax[6].plot(t, log1["I_batt"], label ="Without ECU");ax[6].set_xlabel("time (s)"); ax[6].grid(True)
    ax[6].set_ylabel("A");ax[6].legend()
    fig.tight_layout()
    fig.savefig(fname, dpi=110)
    print("saved", fname)


filename = "uddscol.txt"

BATTERY_MODEL = "spm"  # "spm" = SPPy single particle model, "ecm" = original Thevenin model
BATTERY_KW = dict(     # SPM pack: Ns x Np cells of the 1.65 Ah Gao-Randall-Han set
    Ns=96, Np=30,      #   96s30p = 49.5 Ah, ~384 V at 90 % SoC
    soc0=0.9,          #   start below 100 % so regen has voltage headroom
    R_extra=0.0,       #   extra pack series resistance [Ohm] (busbars, contactors)
    isothermal=True)   #   False = SPPy lumped cell thermal model
CAL_FILE = None        # calibrate on a different cycle (recommended); None = same cycle

START_TIME = 0         # evaluation window [s]
MAX_TIME = None        # integer seconds, or None for the full cycle

TORQUE_SCALE = 0.8     # scale on the baseline torque-demand envelope (0 = off)
DPDT_FRAC = 0.5        # dP/dt limit as a fraction of baseline peak (0 = off)
                       # -> choose with sweep_power_rate.py
FALL_RATIO = 2.0       # lift-off may fall this much faster than it may rise

if __name__ == "__main__":
    t_full, v_full = load_cycle(filename)
    t_cal, v_cal = load_cycle(CAL_FILE) if CAL_FILE else (t_full, v_full)

    # ---- calibrate the ECU from a baseline run (no hard-coded table)
    cal_log = simulate(t_cal, v_cal, False)
    ecu_cfg, ref_dpdt = calibrate_ecu(t_cal, cal_log, TORQUE_SCALE, DPDT_FRAC,
                                      FALL_RATIO)
    print(f"Calibration: baseline peak |dP/dt| = {ref_dpdt / 1e3:.1f} kW/s, "
          f"limit = {DPDT_FRAC * ref_dpdt / 1e3:.1f} kW/s, "
          f"torque scale = {TORQUE_SCALE}\n")

    # ---- evaluation window
    i0 = int(START_TIME * 10)
    i1 = None if MAX_TIME is None else int(MAX_TIME * 10)
    t, v_ref = t_full[i0:i1], v_full[i0:i1]
    if v_ref[0] > 0.1:
        print(f"WARNING: v_ref[0] = {v_ref[0]:.2f} m/s but the vehicle starts at 0; "
              "the first seconds will be a start-up transient.")
    log = simulate(t, v_ref, True, **ecu_cfg)
    log1 = simulate(t, v_ref, False)
    if BATTERY_MODEL == "spm":
        vc = log["V_terminal"] / BATTERY_KW["Ns"]
        print(f"SPM cell voltage with ECU: {vc.min():.3f} .. {vc.max():.3f} V "
              f"(SPM limits 2.5 .. 4.2 V; no charge limiting is modelled)\n")
    summarize(t, log)
    summarize(t, log1)
    compare(metrics(t, log), metrics(t, log1))
    top_didt_events(t, log1)
    plot(t, log, log1, "with_ecu.png")