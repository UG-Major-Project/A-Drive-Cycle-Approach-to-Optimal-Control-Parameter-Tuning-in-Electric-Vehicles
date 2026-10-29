"""
Sweep the ThrottleRateLimiter rise rate and plot the trade-off between
battery-current smoothness and speed tracking.

Usage:
    python sweep_limiter.py                          # uddscol.txt, full cycle
    python sweep_limiter.py --cycle synthetic        # built-in test cycle
    python sweep_limiter.py --start 0 --end 300      # time window in seconds
    python sweep_limiter.py --rise 0.05 0.1 0.2 0.5 1.0 --rmse-tol 0.15
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import main as ev


def run_sweep(t, v_ref, rises, fall_min, hf_scale, v_knee, v_blend):
    base = ev.metrics(t, ev.simulate(t, v_ref, False))
    rows = []
    for r in rises:
        kw = dict(rise_per_s=r,
                  fall_per_s=max(fall_min, 2.0 * r),   # lift-off stays faster than tip-in
                  hf_scale=hf_scale, v_knee=v_knee, v_blend=v_blend)
        rows.append((r, ev.metrics(t, ev.simulate(t, v_ref, True, limiter_kwargs=kw))))
    return base, rows


def print_table(base, rows):
    hdr = f"{'rise [/s]':>10s}{'peak I [A]':>12s}{'RMS I [A]':>11s}" \
          f"{'max dI/dt':>11s}{'RMSE [m/s]':>12s}{'Wh/km':>9s}"
    print(hdr)
    print(f"{'baseline':>10s}{base['peak_I']:>12.1f}{base['rms_I']:>11.2f}"
          f"{base['max_dIdt']:>11.0f}{base['rmse']:>12.3f}{base['Wh_km']:>9.1f}")
    for r, m in rows:
        print(f"{r:>10.3g}{m['peak_I']:>12.1f}{m['rms_I']:>11.2f}"
              f"{m['max_dIdt']:>11.0f}{m['rmse']:>12.3f}{m['Wh_km']:>9.1f}")


def pick_knee(base, rows, rmse_tol):
    """Smallest rise rate (most smoothing) whose RMSE penalty is within tolerance
    AND that actually lowers peak current or max dI/dt by at least 1%."""
    ok = [(r, m) for r, m in rows
          if m["rmse"] - base["rmse"] <= rmse_tol
          and (m["peak_I"] <= 0.99 * base["peak_I"]
               or m["max_dIdt"] <= 0.99 * base["max_dIdt"])]
    if not ok:
        return None
    return min(ok, key=lambda x: x[0])


def plot_pareto(base, rows, fname):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for a, key, lab in ((ax[0], "peak_I", "peak |I_batt| [A]"),
                        (ax[1], "max_dIdt", "max |dI/dt| [A/s]")):
        a.plot([m["rmse"] for _, m in rows], [m[key] for _, m in rows], "o-")
        for r, m in rows:
            a.annotate(f"{r:g}", (m["rmse"], m[key]), textcoords="offset points",
                       xytext=(4, 4), fontsize=8)
        a.plot(base["rmse"], base[key], "r*", ms=12, label="no limiter")
        a.set_xlabel("speed RMSE [m/s]"); a.set_ylabel(lab)
        a.grid(True); a.legend()
    fig.suptitle("Limiter trade-off (labels = rise_per_s)")
    fig.tight_layout()
    fig.savefig(fname, dpi=110)
    print("saved", fname)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cycle", default="uddscol.txt",
                   help="cycle file (time, mph) or 'synthetic'")
    p.add_argument("--start", type=float, default=0.0, help="window start [s]")
    p.add_argument("--end", type=float, default=None, help="window end [s]")
    p.add_argument("--rise", type=float, nargs="+",
                   default=[0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0],
                   help="rise rates to sweep [pedal units / s]")
    p.add_argument("--fall", type=float, default=1.0,
                   help="minimum fall rate [/s]; actual = max(this, 2*rise)")
    p.add_argument("--hf-scale", type=float, default=0.6)
    p.add_argument("--v-knee", type=float, default=15.0)
    p.add_argument("--v-blend", type=float, default=5.0)
    p.add_argument("--rmse-tol", type=float, default=0.2,
                   help="max allowed RMSE increase vs baseline [m/s]")
    p.add_argument("--battery", choices=["spm", "ecm"], default=None,
                   help="override main.BATTERY_MODEL (spm is ~10 s per full-cycle run)")
    p.add_argument("--out", default="sweep_pareto.png")
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

    base, rows = run_sweep(t, v_ref, sorted(a.rise), a.fall,
                           a.hf_scale, a.v_knee, a.v_blend)
    print_table(base, rows)
    pick = pick_knee(base, rows, a.rmse_tol)
    if pick:
        r, m = pick
        print(f"\nSuggested rise_per_s = {r:g}: peak I {m['peak_I']:.1f} A "
              f"(vs {base['peak_I']:.1f}), RMSE +{m['rmse'] - base['rmse']:.3f} m/s")
    else:
        print(f"\nNo rise rate both stayed within {a.rmse_tol} m/s extra RMSE and "
              "reduced peak current or max dI/dt by >=1%.")
    plot_pareto(base, rows, a.out)


if __name__ == "__main__":
    main()