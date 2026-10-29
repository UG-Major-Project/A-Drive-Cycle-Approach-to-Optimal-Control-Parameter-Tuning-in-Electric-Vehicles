import numpy as np


class ThrottleRateLimiter:
    """
    Asymmetric slew-rate limiter for the pedal command u in [-1, 1].

    - Rates are in pedal-units per SECOND (multiplied by dt internally), so
      changing the simulation step does not change the behaviour.
    - The rise limit applies to positive (tip-in) torque only. Brake/regen
      commands (u < 0) pass through undelayed, and releasing the brake is not
      treated as a "rise".
    - The rise rate is scaled smoothly with speed: full rate below v_knee,
      tapering to hf_scale * rise at v_knee + v_blend and above.
    - The fall rate (lift-off) is not speed-dependent, by design.
    """

    def __init__(self, rise_per_s=0.5, fall_per_s=1.0,
                 v_knee=15.0, v_blend=5.0, hf_scale=0.6, dt=0.1):
        self.rise = rise_per_s
        self.fall = fall_per_s
        self.v_knee = v_knee
        self.v_blend = max(v_blend, 1e-6)
        self.hf_scale = hf_scale
        self.dt = dt
        self.u_prev = 0.0

    def reset(self):
        self.u_prev = 0.0

    def apply_throttle_curve(self, u, v):
        """
        u -> raw pedal command [-1, 1]
        v -> vehicle speed in m/s (the unit-delayed speed)
        """
        if u < 0.0:                       # braking/regen: never delay it
            self.u_prev = 0.0
            return u

        # smooth speed-dependent scaling of the rise rate
        s = np.interp(v, [self.v_knee, self.v_knee + self.v_blend],
                         [1.0, self.hf_scale])
        max_up = self.rise * s * self.dt
        max_dn = self.fall * self.dt

        du = min(max(u - self.u_prev, -max_dn), max_up)
        self.u_prev += du
        return self.u_prev


class PowerRateLimiter:
    """
    Limits the rate of change of mechanical power demand, dP/dt = omega * dT/dt,
    by limiting the torque slew rate to  dpdt / omega.

    Battery current follows power, so this is the limit that targets dI/dt:
    the same torque step is allowed to be large at low speed and small at high
    speed. It acts on SIGNED torque, so a tip-in -> regen transition ramps
    through zero instead of jumping (a hard reset to zero at brake onset was
    the dominant remaining dI/dt spike).

    dpdt_up     : max power rise rate [W/s]
    fall_ratio  : lift-off may fall fall_ratio times faster than it may rise
    T_rate_max  : torque-rate ceiling [Nm/s], used near standstill where
                  dpdt/omega would blow up
    omega_floor : lowest wheel-side motor speed used in dpdt/omega [rad/s]
    """

    def __init__(self, dpdt_up=50e3, fall_ratio=2.0, T_rate_max=750.0,
                 omega_floor=30.0, r_wheel=0.30, G=9.0, dt=0.1):
        self.dpdt_up = dpdt_up
        self.fall_ratio = fall_ratio
        self.T_rate_max = T_rate_max
        self.omega_floor = omega_floor
        self.r, self.G, self.dt = r_wheel, G, dt
        self.T_prev = 0.0

    def reset(self):
        self.T_prev = 0.0

    def apply(self, T_req, v):
        omega = max(v / self.r * self.G, self.omega_floor)
        rate_up = min(self.dpdt_up / omega, self.T_rate_max)
        rate_dn = self.fall_ratio * rate_up
        dT = min(max(T_req - self.T_prev, -rate_dn * self.dt), rate_up * self.dt)
        self.T_prev += dT
        return self.T_prev


class TorqueMap:
    """
    1D speed-indexed positive-torque ceiling: T_max(v), linearly interpolated.
    Applies to tip-in torque only; regen/brake torque is never capped here.
    `scale` multiplies the table. An optional floor (resistive torque plus a
    margin) is applied AFTER scaling so the cap can never starve cruising.
    """

    def __init__(self, v_bp, T_max_bp, scale=1.0, T_floor_bp=None):
        self.v_bp = np.asarray(v_bp, dtype=float)
        self.T_bp = np.asarray(T_max_bp, dtype=float)
        self.scale = scale
        self.T_floor_bp = None if T_floor_bp is None else np.asarray(T_floor_bp, float)

    def cap(self, v):
        c = self.scale * float(np.interp(v, self.v_bp, self.T_bp))
        if self.T_floor_bp is not None:
            c = max(c, float(np.interp(v, self.v_bp, self.T_floor_bp)))
        return max(c, 0.0)

    @classmethod
    def from_envelope(cls, v, T_req, bin_w=2.5, scale=1.0, T_floor=20.0,
                      resist_margin=1.5, m=1500.0, Cd=0.28, A=2.2, rho=1.225,
                      Crr=0.01, g=9.81, r_wheel=0.30, G=9.0):
        """Table = max positive torque demanded per speed bin in a baseline run."""
        v, T_req = np.asarray(v), np.asarray(T_req)
        edges = np.arange(0.0, v.max() + bin_w, bin_w)
        centers = 0.5 * (edges[:-1] + edges[1:])
        env = np.full(len(centers), np.nan)
        for i in range(len(centers)):
            sel = (v >= edges[i]) & (v < edges[i + 1]) & (T_req > 0)
            if sel.any():
                env[i] = T_req[sel].max()
        ok = ~np.isnan(env)
        if not ok.any():
            raise ValueError("baseline run has no positive torque demand")
        env = np.interp(centers, centers[ok], env[ok])   # fill unvisited bins
        F_res = m * g * Crr + 0.5 * rho * Cd * A * centers ** 2
        floor = resist_margin * F_res * r_wheel / G
        return cls(centers, np.maximum(env, T_floor), scale, T_floor_bp=floor)

    @classmethod
    def constant_power(cls, P_cap_W, Tpeak=375.0, r_wheel=0.30, G=9.0,
                       v_max=45.0, n=91):
        """T_max(v) = min(Tpeak, P_cap / omega): a mechanical-power ceiling."""
        v = np.linspace(0.1, v_max, n)
        omega = v / r_wheel * G
        return cls(v, np.minimum(Tpeak, P_cap_W / omega))