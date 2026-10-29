"""
Sweep the PowerRateLimiter (dP/dt limit) and plot peak current, max dI/dt and
speed RMSE. By default the torque map is ON, so you see the combined ECU.

The limit is expressed as a fraction of the baseline's own peak |dP/dt|
(measured from a no-ECU run on the calibration cycle), so nothing is hard-coded.

Usage:
    python sweep_power_rate.py                               # uddscol.txt, full cycle
    python sweep_power_rate.py --cycle synthetic
    python sweep_power_rate.py --torque-scale 0              # limiter alone
    python sweep_power_rate.py --frac 0.05 0.1 0.2 0.4 0.7 1.0 --rmse-tol 0.3
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import main as ev


def run_sweep(t, v_ref, fracs, torque_scale, fall_ratio, bin_w, kb):
    base_log = ev.simulate(t, v_ref, False)
    base = ev.metrics(t, base_log)

    cfg0, ref = ev.calibrate_ecu(t, base_log, torque_scale, 0.0, fall_ratio, bin_w)
    map_only = None
    if torque_scale:
        map_only = ev.metrics(t, ev.simulate(t, v_ref, True, Kb_aw=kb, **cfg0))

    rows = []
    for f in fracs:
        cfg, _ = ev.calibrate_ecu(t, base_log, torque_scale, f, fall_ratio, bin_w)
        rows.append((f, ev.metrics(t, ev.simulate(t, v_ref, True, Kb_aw=kb, **cfg))))
    return base, map_only, rows, ref


def print_table(base, map_only, rows, ref):
    print(f"{'dP/dt frac':>11s}{'kW/s':>8s}{'peak I [A]':>12s}{'RMS I [A]':>11s}"
          f"{'max dI/dt':>11s}{'RMSE [m/s]':>12s}{'Wh/km':>9s}")

    def line(lab, kws, m):
        print(f"{lab:>11s}{kws:>8s}{m['peak_I']:>12.1f}{m['rms_I']:>11.2f}"
              f"{m['max_dIdt']:>11.0f}{m['rmse']:>12.3f}{m['Wh_km']:>9.1f}")
    line("baseline", "-", base)
    if map_only is not None:
        line("map only", "-", map_only)
    for f, m in rows:
        line(f"{f:g}", f"{f * ref / 1e3:.1f}", m)


def pick(base, map_only, rows, rmse_tol):
    """
    Among fractions whose EXTRA RMSE (relative to map-only when the map is on,
    else to the no-ECU baseline) is within rmse_tol, the one with the lowest
    max dI/dt, provided it beats the reference by >=1%.
    """
    ref = map_only if map_only is not None else base
    ok = [(f, m) for f, m in rows if m["rmse"] - ref["rmse"] <= rmse_tol]
    if not ok:
        return None
    f, m = min(ok, key=lambda x: x[1]["max_dIdt"])
    return (f, m) if m["max_dIdt"] <= 0.99 * ref["max_dIdt"] else None


def plot_pareto(base, map_only, rows, fname):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for a, key, lab in ((ax[0], "peak_I", "peak |I_batt| [A]"),
                        (ax[1], "max_dIdt", "max |dI/dt| [A/s]")):
        a.plot([m["rmse"] for _, m in rows], [m[key] for _, m in rows], "o-")
        for f, m in rows:
            a.annotate(f"{f:g}", (m["rmse"], m[key]), textcoords="offset points",
                       xytext=(4, 4), fontsize=8)
        a.plot(base["rmse"], base[key], "r*", ms=12, label="no ECU")
        if map_only is not None:
            a.plot(map_only["rmse"], map_only[key], "gs", ms=8, label="map only")
        a.set_xlabel("speed RMSE [m/s]"); a.set_ylabel(lab)
        a.grid(True); a.legend()
    fig.suptitle("Power-rate limiter trade-off (labels = fraction of baseline peak dP/dt)")
    fig.tight_layout()
    fig.savefig(fname, dpi=110)
    print("saved", fname)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cycle", default="uddscol.txt",
                   help="cycle file (time, mph) or 'synthetic'")
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--end", type=float, default=None)
    p.add_argument("--frac", type=float, nargs="+",
                   default=[0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
                   help="dP/dt limit as a fraction of baseline peak |dP/dt|")
    p.add_argument("--torque-scale", type=float, default=0.8,
                   help="torque-map scale; 0 disables the map")
    p.add_argument("--fall-ratio", type=float, default=2.0)
    p.add_argument("--bin-w", type=float, default=2.5)
    p.add_argument("--kb", type=float, default=1.0, help="anti-windup gain")
    p.add_argument("--rmse-tol", type=float, default=0.3,
                   help="max EXTRA RMSE the limiter may add on top of the map "
                        "(or on top of the no-ECU baseline if the map is off) [m/s]")
    p.add_argument("--battery", choices=["spm", "ecm"], default=None,
                   help="override main.BATTERY_MODEL (spm is ~10 s per full-cycle run)")
    p.add_argument("--out", default="sweep_power_rate.png")
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

    base, map_only, rows, ref = run_sweep(t, v_ref, sorted(a.frac), a.torque_scale,
                                          a.fall_ratio, a.bin_w, a.kb)
    print(f"baseline peak |dP/dt|: {ref / 1e3:.1f} kW/s   "
          f"torque scale: {a.torque_scale:g}   fall ratio: {a.fall_ratio:g}\n")
    print_table(base, map_only, rows, ref)

    best = pick(base, map_only, rows, a.rmse_tol)
    if best:
        f, m = best
        print(f"\nSuggested DPDT_FRAC = {f:g} ({f * ref / 1e3:.1f} kW/s): "
              f"max dI/dt {m['max_dIdt']:.0f} (vs {base['max_dIdt']:.0f}), "
              f"peak I {m['peak_I']:.1f} A (vs {base['peak_I']:.1f}), "
              f"RMSE {m['rmse']:.3f} (+{m['rmse'] - base['rmse']:.3f} vs no ECU)")
        print(f"Set  DPDT_FRAC = {f:g}  in main.py")
    else:
        print(f"\nNo fraction added <= {a.rmse_tol} m/s RMSE and cut max dI/dt "
              "by >=1% versus the reference.")
    plot_pareto(base, map_only, rows, a.out)


if __name__ == "__main__":
    main()