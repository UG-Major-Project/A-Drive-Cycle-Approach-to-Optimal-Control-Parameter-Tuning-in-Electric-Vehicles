"""
Sweep a speed-indexed torque ceiling (TorqueMap) and plot peak battery current
and dI/dt against speed-tracking RMSE.

Two ways to build the table (--mode):
  envelope : max torque the baseline run demanded in each speed bin, scaled by s
  power    : constant mechanical-power cap, T_max(v) = s * P_peak / omega

Usage:
    python sweep_torque_map.py                              # uddscol.txt, full cycle
    python sweep_torque_map.py --cycle synthetic
    python sweep_torque_map.py --mode power --scale 0.5 0.6 0.7 0.8 0.9 1.0
    python sweep_torque_map.py --start 0 --end 300 --rmse-tol 0.3
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import main as ev
from ecu_optimizer import TorqueMap


def peak_mech_power(log, r=0.30, G=9.0):
    omega = log["v"] / r * G
    return float(np.max(log["T_actual"] * omega))


def make_map(mode, s, env, P_peak):
    if mode == "envelope":
        return TorqueMap(env.v_bp, env.T_bp, scale=s, T_floor_bp=env.T_floor_bp)
    return TorqueMap.constant_power(s * P_peak)


def run_sweep(t, v_ref, mode, scales, bin_w, kb):
    base_log = ev.simulate(t, v_ref, False)
    base = ev.metrics(t, base_log)
    env = TorqueMap.from_envelope(base_log["v"], base_log["T_req"], bin_w=bin_w)
    P_peak = peak_mech_power(base_log)
    rows = []
    for s in scales:
        tm = make_map(mode, s, env, P_peak)
        log = ev.simulate(t, v_ref, True, use_limiter=False, torque_map=tm, Kb_aw=kb)
        rows.append((s, ev.metrics(t, log), tm))
    return base, rows, env, P_peak


def print_table(base, rows):
    print(f"{'scale':>10s}{'peak I [A]':>12s}{'RMS I [A]':>11s}"
          f"{'max dI/dt':>11s}{'RMSE [m/s]':>12s}{'Wh/km':>9s}")
    print(f"{'baseline':>10s}{base['peak_I']:>12.1f}{base['rms_I']:>11.2f}"
          f"{base['max_dIdt']:>11.0f}{base['rmse']:>12.3f}{base['Wh_km']:>9.1f}")
    for s, m, _ in rows:
        print(f"{s:>10.3g}{m['peak_I']:>12.1f}{m['rms_I']:>11.2f}"
              f"{m['max_dIdt']:>11.0f}{m['rmse']:>12.3f}{m['Wh_km']:>9.1f}")


def pick(base, rows, rmse_tol):
    """Most aggressive cap (smallest scale) within the RMSE budget that helps >=1%."""
    ok = [(s, m, tm) for s, m, tm in rows
          if m["rmse"] - base["rmse"] <= rmse_tol
          and (m["peak_I"] <= 0.99 * base["peak_I"]
               or m["max_dIdt"] <= 0.99 * base["max_dIdt"])]
    return min(ok, key=lambda x: x[0]) if ok else None


def plot_pareto(base, rows, fname):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for a, key, lab in ((ax[0], "peak_I", "peak |I_batt| [A]"),
                        (ax[1], "max_dIdt", "max |dI/dt| [A/s]")):
        a.plot([m["rmse"] for _, m, _ in rows], [m[key] for _, m, _ in rows], "o-")
        for s, m, _ in rows:
            a.annotate(f"{s:g}", (m["rmse"], m[key]), textcoords="offset points",
                       xytext=(4, 4), fontsize=8)
        a.plot(base["rmse"], base[key], "r*", ms=12, label="no cap")
        a.set_xlabel("speed RMSE [m/s]"); a.set_ylabel(lab)
        a.grid(True); a.legend()
    fig.suptitle("Torque-map trade-off (labels = table scale)")
    fig.tight_layout()
    fig.savefig(fname, dpi=110)
    print("saved", fname)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cycle", default="uddscol.txt",
                   help="cycle file (time, mph) or 'synthetic'")
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--end", type=float, default=None)
    p.add_argument("--mode", choices=["envelope", "power"], default="envelope")
    p.add_argument("--scale", type=float, nargs="+",
                   default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    p.add_argument("--bin-w", type=float, default=2.5, help="speed bin width [m/s]")
    p.add_argument("--kb", type=float, default=1.0, help="anti-windup gain")
    p.add_argument("--rmse-tol", type=float, default=0.3,
                   help="max allowed RMSE increase vs baseline [m/s]")
    p.add_argument("--battery", choices=["spm", "ecm"], default=None,
                   help="override main.BATTERY_MODEL (spm is ~10 s per full-cycle run)")
    p.add_argument("--out", default="sweep_torque_map.png")
    a = p.parse_args()
    if a.battery:
        ev.BATTERY_MODEL = a.battery

    if a.cycle == "synthetic":
        t, v_ref = ev.synthetic_cycle()
    else:
        t, v_ref = ev.load_cycle(a.cycle)
    i0 = int(round(a.start / ev.DT))
    i1 = int(round(a.end / ev.DT)) if a.end is not None else len(t)
    t, v_ref = t[i0:i1], v_ref[i0:i1]
    if v_ref[0] > 0.1:
        print(f"WARNING: v_ref[0] = {v_ref[0]:.2f} m/s but the vehicle starts at 0.")

    base, rows, env, P_peak = run_sweep(t, v_ref, a.mode, sorted(a.scale),
                                        a.bin_w, a.kb)
    print(f"baseline peak mechanical power: {P_peak / 1e3:.1f} kW, "
          f"peak torque demand: {env.T_bp.max():.0f} Nm\n")
    print_table(base, rows)

    best = pick(base, rows, a.rmse_tol)
    if best:
        s, m, tm = best
        print(f"\nSuggested scale = {s:g}: peak I {m['peak_I']:.1f} A "
              f"(vs {base['peak_I']:.1f}), max dI/dt {m['max_dIdt']:.0f} "
              f"(vs {base['max_dIdt']:.0f}), RMSE +{m['rmse'] - base['rmse']:.3f} m/s")
        print("Table to paste (v [m/s] -> T_max [Nm]):")
        for v, T in zip(tm.v_bp[::max(1, len(tm.v_bp) // 12)],
                        (tm.scale * tm.T_bp)[::max(1, len(tm.v_bp) // 12)]):
            print(f"  {v:5.1f}  {T:6.1f}")
    else:
        print(f"\nNo scale stayed within {a.rmse_tol} m/s extra RMSE and cut peak "
              "current or max dI/dt by >=1%.")
    plot_pareto(base, rows, a.out)


if __name__ == "__main__":
    main()