"""Command-line report: consumption KPIs, anomaly and potential-leak checks, and forecast evaluation.

Examples:
    python scripts/evaluate.py                                # realistic synthetic demo (app default)
    python scripts/evaluate.py --profile original             # the original v1 generator
    python scripts/evaluate.py --csv sample_data/messy_upload_example.csv
    python scripts/evaluate.py --all-meters --out reports/    # evaluate every meter and write CSV/JSON
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aquasense.core import (REQUIRED, TOTAL, detect_anomalies, episode_scores, evaluate_forecast, flag_scores,  # noqa: E402
                            forecast_summary, generate_data, kpis, meter_summary, potential_leaks, prepare)


def main() -> int:
    warnings.simplefilter("ignore")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv")
    ap.add_argument("--profile", default="realistic", choices=["realistic", "original"])
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--meters", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--all-meters", action="store_true", help="evaluate the forecast for every meter, not just the total")
    ap.add_argument("--out")
    a = ap.parse_args()
    truth = None
    if a.csv:
        data, rep = prepare(pd.read_csv(a.csv, low_memory=False))
        print(rep.as_frame().to_string(index=False), "\n")
    else:
        truth = generate_data(a.days, a.meters, a.seed, truth=True, profile=a.profile)
        data = truth[REQUIRED]
    k = kpis(data)
    print(f"{k['meters']} meters · {k['readings']:,} hourly readings · {k['total_liters'] / 1000:,.1f} m³ · "
          f"night flow {k['night_lph']:.1f} L/h ({k['night_share']:.0%} of average)")

    hourly, ep = potential_leaks(data)
    print(f"\nPotential-leak episodes: {len(ep)} ({(ep.priority == 'Higher').sum()} higher priority)")
    if len(ep):
        print(ep.drop(columns="status").round(2).to_string(index=False))
    an = detect_anomalies(data, 0.01)
    print(f"Unusual readings (1% rate): {int(an.anomaly.sum())}")
    if truth is not None:
        print("\nSynthetic checks against injected events:")
        print("  leak episodes, all:           ", episode_scores(ep, truth))
        print("  leak episodes, higher only:   ", episode_scores(ep[ep.priority == 'Higher'], truth))
        print("  anomaly flags vs leak hours:  ", {k_: round(float(v), 3) if isinstance(v, float) else v
                                                   for k_, v in flag_scores(an, truth, 'anomaly').items()})
        if "injected_burst" in truth:
            b = flag_scores(an, truth.assign(injected_leak=truth.injected_burst), 'anomaly')
            print("  anomaly flags vs burst hours: ", {k_: round(float(v), 3) if isinstance(v, float) else v for k_, v in b.items()})

    series = [TOTAL] + (sorted(data.meter_id.unique()) if a.all_meters else [])
    rows, summaries = [], []
    for s in series:
        try:
            ev = evaluate_forecast(data, s)
        except ValueError as exc:
            print(f"  {s}: skipped ({exc})")
            continue
        d = ev.day_ahead.set_index("method")
        rows.append({"series": s, "rf_day_ahead_mae": d.MAE.iloc[0], "naive_same_hour_mae": d.MAE.iloc[1],
                     "hour_profile_mae": d.MAE.iloc[2], "rf_one_step_mae": ev.one_step.MAE.iloc[0],
                     "origins": ev.meta["origins"]})
        summaries.append(forecast_summary(ev))
    fc = pd.DataFrame(rows)
    print("\nForecast evaluation (MAE, L/h; day-ahead = recursive 24 h from 23:00 in the 20% holdout):")
    print(fc.round(3).to_string(index=False))
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        meter_summary(data).round(4).to_csv(out / "meter_summary.csv", index=False)
        ep.round(3).to_csv(out / "potential_leak_episodes.csv", index=False)
        fc.round(4).to_csv(out / "forecast_evaluation.csv", index=False)
        (out / "forecast_metrics.json").write_text(json.dumps(summaries, indent=2, default=str))
        print(f"\nReports written to {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
