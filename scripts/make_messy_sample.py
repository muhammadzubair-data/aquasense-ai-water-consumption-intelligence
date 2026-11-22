"""Build sample_data/messy_upload_example.csv: a deliberately messy 15-minute export used to demo and test prepare()."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aquasense.core import generate_data  # noqa: E402


def build() -> pd.DataFrame:
    hourly = generate_data(21, 3, seed=5, profile='realistic')
    rng = np.random.default_rng(9)
    rows = []
    for r in hourly.itertuples():
        split = rng.dirichlet(np.ones(4))            # spread each hour over four 15-minute readings
        for q in range(4):
            t = r.timestamp + pd.Timedelta(minutes=15 * q)
            rows.append((t.strftime('%Y-%m-%dT%H:%M:%S+05:00'), r.meter_id.replace('METER', 'WM'), r.liters * split[q] / 1000))
    d = pd.DataFrame(rows, columns=['Reading Time', 'Meter No', 'Consumption m3'])
    d['Consumption m3'] = d['Consumption m3'].round(6).astype(object)
    d['Customer Note'] = ''
    d = d.drop(index=range(4 * 30, 4 * 32)).reset_index(drop=True)          # 2-hour gap on WM-001 (interpolated)
    d = d.drop(index=range(4 * 200, 4 * 212)).reset_index(drop=True)        # 12-hour gap on WM-001 (left missing)
    d.loc[5:7, 'Consumption m3'] = ['n/a', '', '-0.004']                    # bad and negative volumes
    d.loc[8, 'Reading Time'] = 'not a date'
    d.loc[9, 'Meter No'] = ''
    d = pd.concat([d, d.iloc[100:110]])                                      # exact duplicates
    conflict = d.iloc[[300, 301]].copy()
    conflict['Consumption m3'] = conflict['Consumption m3'].astype(float) * 1.5
    d = pd.concat([d, conflict])                                             # conflicting duplicates
    daily = pd.DataFrame({'Reading Time': pd.date_range('2026-01-01', periods=21, freq='D').strftime('%Y-%m-%dT00:00:00+05:00'),
                          'Meter No': 'WM-DAILY', 'Consumption m3': 0.9, 'Customer Note': ''})
    return pd.concat([d, daily], ignore_index=True)


if __name__ == '__main__':
    out = Path(__file__).resolve().parents[1] / 'sample_data' / 'messy_upload_example.csv'
    build().to_csv(out, index=False)
    print(out, len(build()))
