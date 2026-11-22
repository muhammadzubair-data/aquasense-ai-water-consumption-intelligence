"""AquaSense AI analytics core.

Synthetic water-meter data, forgiving upload preparation, consumption analytics, anomaly detection, potential-leak
indicators and leakage-safe demand forecasting. Everything here is a demonstration on synthetic data: anomaly flags
and potential-leak indicators are statistical signals, not confirmed leaks.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

REQUIRED = ['timestamp', 'meter_id', 'liters']
FEATURES = ['hour', 'weekday', 'lag1', 'lag24', 'rolling24']      # the original forecasting features
FEATURE_LABELS = {'hour': 'Hour of day', 'weekday': 'Day of week', 'lag1': 'Previous hour',
                  'lag24': 'Same hour yesterday', 'rolling24': 'Trailing 24-hour mean'}
HORIZON = 24
TEST_SHARE = 0.2
SEED = 42
NIGHT_HOURS = (2, 3, 4)
TOTAL = 'ALL METERS'
MAX_ROWS = 500_000
MAX_METERS = 200
MIN_MODEL_HOURS = 96
METER_RE = re.compile(r'^[\w\-.:/# ]{1,64}$')

ALIASES = {
    'time': 'timestamp', 'datetime': 'timestamp', 'date_time': 'timestamp', 'reading_time': 'timestamp',
    'read_time': 'timestamp', 'ts': 'timestamp', 'date': 'timestamp', 'interval_start': 'timestamp',
    'meter': 'meter_id', 'meterid': 'meter_id', 'meter_no': 'meter_id', 'meter_number': 'meter_id',
    'device_id': 'meter_id', 'device': 'meter_id', 'sensor_id': 'meter_id', 'account_meter': 'meter_id',
    'litres': 'liters', 'liter': 'liters', 'litre': 'liters', 'l': 'liters', 'consumption_l': 'liters',
    'consumption_liters': 'liters', 'consumption_litres': 'liters', 'volume_l': 'liters', 'usage_l': 'liters',
    'water_l': 'liters', 'consumption': 'liters', 'usage': 'liters', 'volume': 'liters', 'flow_l': 'liters',
}
UNIT_ALIASES = {  # column name → (multiplier to liters, label)
    'm3': (1000.0, 'cubic metres'), 'consumption_m3': (1000.0, 'cubic metres'), 'volume_m3': (1000.0, 'cubic metres'),
    'usage_m3': (1000.0, 'cubic metres'), 'gallons': (3.785411784, 'US gallons'), 'gal': (3.785411784, 'US gallons'),
    'consumption_gallons': (3.785411784, 'US gallons'), 'usage_gallons': (3.785411784, 'US gallons'),
}


# ------------------------------------------------------------------------------------------------ data
# Realistic residential diurnal shape (mean 1.0): low overnight flow, morning and evening peaks.
REALISTIC_SHAPE = np.array([.32, .24, .20, .19, .22, .45, 1.15, 1.75, 1.70, 1.25, 1.00, .95,
                            1.05, 1.00, .90, .88, .98, 1.25, 1.60, 1.72, 1.55, 1.25, .85, .50])
REALISTIC_SHAPE = REALISTIC_SHAPE / REALISTIC_SHAPE.mean()
PROFILES = ('original', 'realistic')


def generate_data(days: int = 90, meters: int = 8, seed: int = SEED, truth: bool = False,
                  profile: str = 'original') -> pd.DataFrame:
    """Synthetic hourly readings.

    ``profile='original'`` reproduces the original row-by-row generator exactly (tested). ``profile='realistic'`` keeps
    the same random draws (so meters, leak positions and noise line up) but applies a residential daily shape with low
    night flow, a later weekend morning peak, and 1–3 short legitimate high-use bursts per meter (e.g. irrigation or
    filling a tank) from a separate random stream. Every fourth meter gets a 40-hour +27 L/h leak in both profiles.
    With ``truth=True`` the columns ``injected_leak`` (and ``injected_burst`` for the realistic profile) are added."""
    if profile not in PROFILES:
        raise ValueError(f'profile must be one of {PROFILES}')
    rng = np.random.default_rng(seed)
    ts = pd.date_range('2026-01-01', periods=days * 24, freq='h')
    n = len(ts)
    hour = ts.hour.to_numpy()
    shape = 13 * np.sin((hour - 7) * np.pi / 12) ** 2
    peak = np.where(((hour >= 6) & (hour <= 9)) | ((hour >= 18) & (hour <= 22)), 8, 0)
    weekend = ts.dayofweek.to_numpy() >= 5
    real = np.where(weekend, REALISTIC_SHAPE[(hour - 2) % 24] * .55 + REALISTIC_SHAPE[hour] * .55, REALISTIC_SHAPE[hour])
    idx = np.arange(n)
    r2 = np.random.default_rng(seed + 101)
    parts = []
    for m in range(meters):
        baseline = rng.uniform(18, 42)
        leak_start = int(rng.integers(n // 2, n - 48)) if n > 120 else n + 1
        noise = rng.normal(0, 3.2, n)
        leak = (m % 4 == 0) & (idx >= leak_start) & (idx < leak_start + 40)
        burst = np.zeros(n, dtype=bool)
        if profile == 'original':
            demand = baseline + shape + peak + noise
        else:
            demand = baseline * real + noise * (.35 + .45 * real)
            extra = np.zeros(n)
            for _ in range(int(r2.integers(1, 4))):
                for _attempt in range(20):
                    day = int(r2.integers(1, max(2, days - 1)))
                    start = day * 24 + int(r2.integers(8, 19))
                    length = int(r2.integers(2, 5))
                    if start + length < n and not leak[max(0, start - 48):start + length + 48].any() \
                            and not burst[max(0, start - 24):start + length + 24].any():
                        burst[start:start + length] = True
                        extra[start:start + length] = r2.uniform(35, 60)
                        break
            demand = demand + extra
        demand = np.where(leak, demand + 27, demand)
        part = pd.DataFrame({'timestamp': ts, 'meter_id': f'METER-{m + 1:03d}',
                             'liters': np.round(np.maximum(0, demand), 2)})
        if truth:
            part['injected_leak'] = leak
            if profile == 'realistic':
                part['injected_burst'] = burst
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Strict validation (original behaviour): raises ValueError on any problem."""
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {", ".join(sorted(missing))}')
    out = df[REQUIRED].copy()
    out['timestamp'] = pd.to_datetime(out['timestamp'], errors='coerce')
    out['liters'] = pd.to_numeric(out['liters'], errors='coerce')
    if out[REQUIRED].isna().any().any():
        raise ValueError('Missing or invalid timestamps, meter IDs or liters')
    if (out['liters'] < 0).any():
        raise ValueError('Liters cannot be negative')
    if out.duplicated(['timestamp', 'meter_id']).any():
        raise ValueError('Duplicate meter/timestamp pairs')
    return out.sort_values(['meter_id', 'timestamp']).reset_index(drop=True)


@dataclass
class DataReport:
    rows_in: int = 0
    rows_out: int = 0
    renamed: dict = field(default_factory=dict)
    ignored_columns: list = field(default_factory=list)
    unit_note: str = ''
    dropped_timestamp: int = 0
    dropped_meter: int = 0
    dropped_liters: int = 0
    dropped_negative: int = 0
    exact_duplicates: int = 0
    conflicting_duplicates: int = 0
    resampled_meters: int = 0
    excluded_meters: list = field(default_factory=list)
    interpolated_hours: int = 0
    missing_hours: int = 0
    meters: int = 0
    short_meters: int = 0
    first: str = ''
    last: str = ''
    notes: list = field(default_factory=list)

    def as_frame(self) -> pd.DataFrame:
        rows = [('Rows in file', self.rows_in), ('Hourly readings kept', self.rows_out), ('Meters', self.meters),
                ('Period', f'{self.first} → {self.last}' if self.first else '—'),
                ('Dropped: unparseable timestamp', self.dropped_timestamp),
                ('Dropped: missing or invalid meter ID', self.dropped_meter),
                ('Dropped: missing or non-numeric volume', self.dropped_liters),
                ('Dropped: negative volume', self.dropped_negative),
                ('Removed: exact duplicate rows', self.exact_duplicates),
                ('Merged: conflicting duplicates (same meter and time, mean kept)', self.conflicting_duplicates),
                ('Meters resampled from sub-hourly to hourly totals', self.resampled_meters),
                ('Meters excluded (readings coarser than hourly)', ', '.join(self.excluded_meters) or '—'),
                ('Short gaps filled by interpolation (≤ 3 h)', self.interpolated_hours),
                ('Hours still missing (longer gaps)', self.missing_hours),
                (f'Meters under {MIN_MODEL_HOURS} hours (analytics only, no models)', self.short_meters),
                ('Columns renamed', ', '.join(f'{a} → {b}' for a, b in self.renamed.items()) or '—'),
                ('Unit conversion', self.unit_note or 'none (litres)'),
                ('Ignored columns', ', '.join(self.ignored_columns) or '—')]
        return pd.DataFrame(rows, columns=['check', 'result']).astype({'result': str})


def _norm(c) -> str:
    return re.sub(r'[^0-9a-z]+', '_', str(c).strip().lower()).strip('_')


def _parse_times(s: pd.Series) -> tuple[pd.Series, str]:
    """Parse timestamps keeping local wall-clock time (water use follows the local day). A single UTC offset is
    dropped without shifting; mixed offsets (e.g. a daylight-saving change) are converted to UTC, with a note."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            t = pd.to_datetime(s, errors='coerce', format='mixed')
    except (ValueError, TypeError):                              # pandas 3: mixed offsets raise
        t = pd.Series(np.array([None] * len(s), dtype=object), index=s.index)
    if isinstance(t.dtype, pd.DatetimeTZDtype):
        return t.dt.tz_localize(None), 'Time-zone offset removed; local clock times kept.'
    if t.dtype == object:                                       # mixed offsets → no single local clock
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            u = pd.to_datetime(s, errors='coerce', utc=True, format='mixed')
        return u.dt.tz_convert(None), 'Mixed time-zone offsets found; timestamps converted to UTC.'
    return t, ''


def prepare(raw: pd.DataFrame, max_rows: int = MAX_ROWS, max_meters: int = MAX_METERS) -> tuple[pd.DataFrame, DataReport]:
    """Forgiving validation for uploads. Returns an hourly frame ``timestamp, meter_id, liters`` (sorted, one row per
    meter and hour, gaps over 3 hours left out) and a report of every repair. Raises ValueError only when nothing
    usable remains."""
    rep = DataReport(rows_in=len(raw))
    if len(raw) > max_rows:
        raise ValueError(f'The file has {len(raw):,} rows; the limit for this demo is {max_rows:,}.')
    cols, used, factor = {}, set(), 1.0
    for c in raw.columns:
        n = _norm(c)
        target = n if n in REQUIRED else ALIASES.get(n)
        if target is None and n in UNIT_ALIASES and 'liters' not in used:
            target = 'liters'
            factor, label = UNIT_ALIASES[n]
            rep.unit_note = f'{c} converted from {label} to litres (× {factor:g})'
        if target and target not in used:
            cols[c] = target
            used.add(target)
            if n != target:
                rep.renamed[str(c)] = target
        else:
            rep.ignored_columns.append(str(c))
    missing = [c for c in REQUIRED if c not in used]
    if missing:
        raise ValueError('Missing required columns: ' + ', '.join(missing) +
                         '. Expected timestamp, meter_id and liters (common alternatives such as datetime, '
                         'meter, litres or m3 are recognised).')
    d = raw[list(cols)].rename(columns=cols).copy()

    d['timestamp'], tz_note = _parse_times(d['timestamp'])
    if tz_note:
        rep.notes.append(tz_note)
    bad_t = d['timestamp'].isna()
    rep.dropped_timestamp = int(bad_t.sum())
    d = d[~bad_t]

    mid = d['meter_id'].astype('string').str.strip()
    bad_m = mid.isna() | ~mid.fillna('').str.match(METER_RE) | mid.isin(['nan', 'None', 'NaN', 'null'])
    rep.dropped_meter = int(bad_m.sum())
    d = d.assign(meter_id=mid)[~bad_m]

    v = pd.to_numeric(d['liters'], errors='coerce') * factor
    bad_v = v.isna() | ~np.isfinite(v.astype(float))
    rep.dropped_liters = int(bad_v.sum())
    d = d.assign(liters=v)[~bad_v]
    neg = d['liters'] < 0
    rep.dropped_negative = int(neg.sum())
    d = d[~neg]

    before = len(d)
    d = d.drop_duplicates()
    rep.exact_duplicates = before - len(d)
    dup = d.duplicated(['meter_id', 'timestamp'], keep=False)
    rep.conflicting_duplicates = int(d[dup].groupby(['meter_id', 'timestamp']).ngroups)
    d = d.groupby(['meter_id', 'timestamp'], as_index=False)['liters'].mean()

    if d.empty:
        raise ValueError('No valid readings remain after cleaning (check timestamps, meter IDs and volumes).')
    if d['meter_id'].nunique() > max_meters:
        raise ValueError(f'{d.meter_id.nunique()} meters found; the limit for this demo is {max_meters}.')

    parts = []
    for meter, g in d.groupby('meter_id', sort=True):
        g = g.sort_values('timestamp')
        step = g['timestamp'].diff().median() if len(g) > 1 else pd.Timedelta(hours=1)
        if pd.isna(step) or step > pd.Timedelta(minutes=90):
            rep.excluded_meters.append(str(meter))
            continue
        if step < pd.Timedelta(minutes=55):
            rep.resampled_meters += 1
        h = g.assign(timestamp=g['timestamp'].dt.floor('h')).groupby('timestamp')['liters'].sum()
        full = pd.date_range(h.index.min(), h.index.max(), freq='h')
        h = h.reindex(full)
        gap = h.isna()
        filled = h.interpolate(limit=3, limit_area='inside')
        # only fill runs of 3 hours or fewer (interpolate(limit) would partially fill longer runs)
        run_id = (gap != gap.shift()).cumsum()
        run_len = gap.groupby(run_id).transform('sum')
        fill_ok = gap & (run_len <= 3)
        h = h.where(~fill_ok, filled)
        rep.interpolated_hours += int(fill_ok.sum())
        rep.missing_hours += int((gap & ~fill_ok).sum())
        parts.append(pd.DataFrame({'timestamp': h.index, 'meter_id': str(meter), 'liters': h.to_numpy()}).dropna())
    if not parts:
        raise ValueError('No meter has hourly or finer readings. AquaSense needs hourly (or more frequent) data.')
    out = pd.concat(parts, ignore_index=True).sort_values(['meter_id', 'timestamp']).reset_index(drop=True)
    out['liters'] = out['liters'].round(3)
    hours = out.groupby('meter_id').size()
    rep.rows_out, rep.meters = len(out), int(out.meter_id.nunique())
    rep.short_meters = int((hours < MIN_MODEL_HOURS).sum())
    rep.first, rep.last = str(out.timestamp.min()), str(out.timestamp.max())
    if rep.resampled_meters:
        rep.notes.append('Sub-hourly readings were summed into hourly totals (assumes interval volumes, not meter totals).')
    if rep.excluded_meters:
        rep.notes.append('Meters with daily or coarser readings were left out: hourly patterns cannot be analysed.')
    if hours.max() < 48:
        raise ValueError('Every meter has fewer than 48 hourly readings; at least two days are needed.')
    return out, rep


def grid(df: pd.DataFrame, meter: str) -> pd.Series:
    """Hourly series for one meter (or the network total) on a regular grid; missing hours are NaN."""
    if meter == TOTAL:
        wide = df.pivot_table(index='timestamp', columns='meter_id', values='liters', aggfunc='sum')
        s = wide.sum(axis=1, min_count=wide.shape[1])          # NaN when any meter is missing that hour
    else:
        s = df.loc[df.meter_id == meter].set_index('timestamp')['liters']
    if s.empty:
        raise ValueError(f'No readings for {meter}.')
    return s.sort_index().asfreq('h').rename('liters')


# ------------------------------------------------------------------------------------------------ analytics
def kpis(df: pd.DataFrame) -> dict:
    t = df.timestamp
    days = max((t.max() - t.min()) / pd.Timedelta(days=1), 1 / 24)
    prof = df.groupby(t.dt.hour)['liters'].mean()
    night = df[t.dt.hour.isin(NIGHT_HOURS)]['liters'].mean()
    daily = df.groupby(t.dt.date)['liters'].sum()
    return {'total_liters': float(df.liters.sum()), 'readings': int(len(df)), 'meters': int(df.meter_id.nunique()),
            'days': float(days), 'first': t.min(), 'last': t.max(), 'mean_lph': float(df.liters.mean()),
            'mean_daily_liters': float(daily.mean()), 'peak_daily_liters': float(daily.max()),
            'peak_day': daily.idxmax(), 'peak_hour': int(prof.idxmax()), 'night_lph': float(night),
            'night_share': float(night / df.liters.mean()) if df.liters.mean() else np.nan}


def daily(df: pd.DataFrame, by_meter: bool = False) -> pd.DataFrame:
    keys = [df.timestamp.dt.normalize().rename('date')] + ([df.meter_id] if by_meter else [])
    return df.groupby(keys)['liters'].agg(liters='sum', hours='size').reset_index()


def hourly_profile(df: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    x = df.assign(hour=df.timestamp.dt.hour, day_type=np.where(df.timestamp.dt.dayofweek >= 5, 'Weekend', 'Weekday'))
    keys = ['hour'] + ([by] if by else [])
    return x.groupby(keys)['liters'].agg(mean='mean', p10=lambda s: s.quantile(.1),
                                         p90=lambda s: s.quantile(.9)).reset_index()


def weekday_hour(df: pd.DataFrame) -> pd.DataFrame:
    net = df.groupby('timestamp')['liters'].sum()
    x = pd.DataFrame({'weekday': net.index.dayofweek, 'hour': net.index.hour, 'liters': net.to_numpy()})
    return x.pivot_table(index='weekday', columns='hour', values='liters', aggfunc='mean')


def peak_periods(df: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    net = df.groupby('timestamp')['liters'].agg(liters='sum', meters='size').reset_index()
    return net.sort_values('liters', ascending=False).head(top).reset_index(drop=True)


def meter_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows, total = [], df.liters.sum()
    for m, g in df.groupby('meter_id'):
        t = g.timestamp
        prof = g.groupby(t.dt.hour)['liters'].mean()
        night = g[t.dt.hour.isin(NIGHT_HOURS)]['liters'].mean()
        last = g[t > t.max() - pd.Timedelta(days=7)]['liters'].mean()
        prev = g[(t <= t.max() - pd.Timedelta(days=7)) & (t > t.max() - pd.Timedelta(days=14))]['liters'].mean()
        rows.append({'meter_id': m, 'hours': len(g), 'total_liters': g.liters.sum(), 'share': g.liters.sum() / total,
                     'mean_lph': g.liters.mean(), 'p95_lph': g.liters.quantile(.95), 'peak_hour': int(prof.idxmax()),
                     'night_lph': night, 'night_ratio': night / g.liters.mean() if g.liters.mean() else np.nan,
                     'load_factor': g.liters.mean() / prof.max() if prof.max() else np.nan,
                     'variability_cv': g.liters.std() / g.liters.mean() if g.liters.mean() else np.nan,
                     'last7_vs_prev7': (last / prev - 1) if prev and not np.isnan(prev) else np.nan})
    return pd.DataFrame(rows).sort_values('total_liters', ascending=False).reset_index(drop=True)


def night_flow(df: pd.DataFrame) -> pd.DataFrame:
    """Mean 02:00–04:59 flow per meter and night, compared with that meter's median night flow."""
    x = df[df.timestamp.dt.hour.isin(NIGHT_HOURS)]
    n = x.groupby([x.meter_id, x.timestamp.dt.normalize().rename('date')])['liters'].mean().rename('night_lph').reset_index()
    n['typical_night_lph'] = n.groupby('meter_id')['night_lph'].transform('median')
    n['ratio'] = n.night_lph / n.typical_night_lph
    return n


# ------------------------------------------------------------------------------------------------ anomalies
def add_anomalies(df, contamination=0.04):
    """Original method: Isolation Forest per meter on liters, hour and weekday."""
    out = validate_data(df)
    out['hour'] = out.timestamp.dt.hour
    out['weekday'] = out.timestamp.dt.dayofweek
    out['anomaly'] = False
    out['anomaly_score'] = 0.0
    for _, part in out.groupby('meter_id'):
        if len(part) < 20:
            continue
        model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
        x = part[['liters', 'hour', 'weekday']]
        out.loc[part.index, 'anomaly'] = model.fit_predict(x) == -1
        out.loc[part.index, 'anomaly_score'] = -model.decision_function(x)
    return out


def expected_same_hour(s: pd.Series, days: int = 14, min_days: int = 3) -> pd.Series:
    """Typical value for each hour from past data only: the median of the same hour on earlier days of the same type
    (weekdays compared with the previous 10 weekdays, weekend days with the previous 4 weekend days), falling back to
    the previous ``days`` days of any type while a day type has too little history."""
    frame = pd.DataFrame({'y': s.to_numpy(), 'hour': s.index.hour,
                          'weekend': s.index.dayofweek >= 5}, index=s.index)
    any_day = frame.groupby('hour')['y'].transform(lambda v: v.shift(1).rolling(days, min_periods=min_days).median())
    wd = frame[~frame.weekend].groupby('hour')['y'].transform(lambda v: v.shift(1).rolling(10, min_periods=min_days).median())
    we = frame[frame.weekend].groupby('hour')['y'].transform(lambda v: v.shift(1).rolling(4, min_periods=min_days).median())
    typed = pd.concat([wd, we]).reindex(frame.index)
    return typed.fillna(any_day).rename('expected')


def detect_anomalies(df: pd.DataFrame, contamination: float = 0.02, meters=None) -> pd.DataFrame:
    """Context-aware Isolation Forest per meter. Features: the reading, its deviation from the typical value for that
    hour (previous 14 days, same hour), and the 3-hour mean of that deviation. Returns readings with ``anomaly``,
    ``anomaly_score``, ``expected`` and ``excess``. The first days of each meter have no baseline and are not scored."""
    out = []
    for m in (meters or sorted(df.meter_id.unique())):
        s = grid(df, m)
        exp = expected_same_hour(s)
        f = pd.DataFrame({'liters': s, 'expected': exp})
        f['excess'] = f.liters - f.expected
        f['excess_3h'] = f['excess'].rolling(3, min_periods=2).mean()
        f = f.dropna()
        if len(f) < 48:
            continue
        model = IsolationForest(n_estimators=200, contamination=contamination, random_state=SEED)
        x = f[['liters', 'excess', 'excess_3h']]
        f['anomaly'] = model.fit_predict(x) == -1
        f['anomaly_score'] = -model.decision_function(x)
        f['meter_id'] = m
        out.append(f.reset_index(names='timestamp'))
    if not out:
        return pd.DataFrame(columns=['timestamp', 'liters', 'expected', 'excess', 'excess_3h', 'anomaly',
                                     'anomaly_score', 'meter_id', 'direction'])
    r = pd.concat(out, ignore_index=True)
    r['direction'] = np.where(r.excess >= 0, 'higher than usual', 'lower than usual')
    return r[['timestamp', 'meter_id', 'liters', 'expected', 'excess', 'excess_3h', 'anomaly', 'anomaly_score',
              'direction']]


def flag_scores(flags: pd.DataFrame, truth: pd.DataFrame, col: str) -> dict:
    """Hour-level agreement between a boolean flag column and injected leak hours (synthetic demo only)."""
    m = flags.merge(truth[['timestamp', 'meter_id', 'injected_leak']], on=['timestamp', 'meter_id'], how='inner')
    f, t = m[col].astype(bool), m['injected_leak'].astype(bool)
    tp = int((f & t).sum())
    return {'flagged_hours': int(f.sum()), 'leak_hours': int(t.sum()), 'flagged_in_leak': tp,
            'precision': tp / f.sum() if f.sum() else np.nan, 'recall': tp / t.sum() if t.sum() else np.nan,
            'scored_hours': int(len(m))}


# ------------------------------------------------------------------------------------------------ potential leaks
def leak_signals(df, window=12, threshold=1.35):
    """Original heuristic: trailing rolling mean vs an expanding all-hours median. Not a leak diagnosis."""
    out = validate_data(df)
    out['leak_signal'] = False
    for _, part in out.groupby('meter_id'):
        if len(part) < window * 3:
            continue
        rolling = part.liters.rolling(window, min_periods=window).mean()
        baseline = part.liters.expanding(min_periods=window).median().shift(window)
        flags = (rolling > baseline * threshold) & (rolling > baseline + 8)
        out.loc[part.index, 'leak_signal'] = flags.fillna(False).to_numpy()
    return out


def potential_leaks(df: pd.DataFrame, window: int = 12, ratio: float = 1.35, min_excess: float = 8.0,
                    min_hours: int = 6, meters=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sustained elevated consumption compared with what is normal *for each hour of the day*.

    For every meter: expected = median of the same hour over the previous 14 days. A reading is part of a signal
    when the trailing ``window``-hour mean is above ``ratio`` × the expected mean for the same hours and at least
    ``min_excess`` L/h above it. Consecutive signal hours (gaps of one hour allowed) form an episode; episodes shorter
    than ``min_hours`` are dropped. These are *potential* indicators only."""
    hourly, episodes = [], []
    for m in (meters or sorted(df.meter_id.unique())):
        s = grid(df, m)
        if s.notna().sum() < 4 * 24:
            continue
        exp = expected_same_hour(s)
        mp = max(2, int(window * .75))
        ro = s.rolling(window, min_periods=mp).mean()
        re_ = exp.rolling(window, min_periods=mp).mean()
        sig = ((ro > ratio * re_) & (ro - re_ > min_excess)).fillna(False)
        f = pd.DataFrame({'liters': s, 'expected': exp, 'rolling_mean': ro, 'rolling_expected': re_, 'signal': sig})
        f['excess'] = (f.liters - f.expected)
        # episodes: join runs separated by a single quiet hour
        run = sig.copy()
        bridged = (~sig) & sig.shift(1, fill_value=False) & sig.shift(-1, fill_value=False)
        run = run | bridged
        rid = (run != run.shift()).cumsum()
        f['episode'] = np.nan
        for _, g in f[run].groupby(rid[run]):
            hours = len(g)
            if hours < min_hours:
                continue
            start, end = g.index.min(), g.index.max()
            night = g[g.index.hour.isin(NIGHT_HOURS)]
            excess = g.excess.clip(lower=0).sum()
            k = len(episodes) + 1
            f.loc[start:end, 'episode'] = k
            night_up = int((night.excess > min_excess).sum())
            priority = 'Higher' if night_up >= 2 and hours >= 12 else 'Lower'
            episodes.append({'episode': k, 'meter_id': m, 'start': start, 'end': end, 'hours': hours,
                             'excess_liters': float(excess), 'mean_excess_lph': float(g.excess.mean()),
                             'peak_ratio': float((g.rolling_mean / g.rolling_expected).max()),
                             'night_hours_elevated': night_up, 'priority': priority,
                             'status': 'Potential indicator · needs on-site check'})
        f['in_episode'] = f.episode.notna()
        f['meter_id'] = m
        hourly.append(f.reset_index(names='timestamp'))
    h = pd.concat(hourly, ignore_index=True) if hourly else pd.DataFrame(
        columns=['timestamp', 'liters', 'expected', 'rolling_mean', 'rolling_expected', 'signal', 'excess', 'episode',
                 'in_episode', 'meter_id'])
    e = pd.DataFrame(episodes, columns=['episode', 'meter_id', 'start', 'end', 'hours', 'excess_liters',
                                        'mean_excess_lph', 'peak_ratio', 'night_hours_elevated', 'priority', 'status'])
    return h, e


def leak_windows(truth: pd.DataFrame) -> pd.DataFrame:
    t = truth[truth.injected_leak]
    if t.empty:
        return pd.DataFrame(columns=['meter_id', 'start', 'end'])
    return t.groupby('meter_id').timestamp.agg(start='min', end='max').reset_index()


def episode_scores(episodes: pd.DataFrame, truth: pd.DataFrame, window: int = 12) -> dict:
    """Episode-level comparison with injected leaks (synthetic demo only). A leak counts as found when an episode
    overlaps it (allowing the rolling window's lag); delay = first episode hour − leak start."""
    w = leak_windows(truth)
    found, delays, used = 0, [], set()
    for r in w.itertuples():
        hit = episodes[(episodes.meter_id == r.meter_id) & (episodes.start <= r.end + pd.Timedelta(hours=window))
                       & (episodes.end >= r.start)]
        if len(hit):
            found += 1
            used.update(hit.episode)
            delays.append((hit.start.min() - r.start) / pd.Timedelta(hours=1))
    return {'injected_leaks': int(len(w)), 'leaks_found': found, 'episodes': int(len(episodes)),
            'false_episodes': int(len(episodes) - len(used)),
            'mean_delay_hours': float(np.mean(delays)) if delays else np.nan}


# ------------------------------------------------------------------------------------------------ forecasting
def features(s: pd.Series) -> pd.DataFrame:
    """Leakage-safe features on a regular hourly grid: every value uses only earlier hours."""
    f = pd.DataFrame({'liters': s})
    f['hour'] = s.index.hour
    f['weekday'] = s.index.dayofweek
    f['lag1'] = s.shift(1)
    f['lag24'] = s.shift(24)
    f['rolling24'] = s.shift(1).rolling(24, min_periods=24).mean()
    return f.dropna()


def make_model(n_estimators: int = 100) -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=3, random_state=SEED, n_jobs=1)


def _metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {'MAE': float(mean_absolute_error(y, p)), 'RMSE': float(np.sqrt(mean_squared_error(y, p))),
            'bias': float(np.mean(p - y)), 'n': int(len(y))}


def _recursive(model, histories: list[np.ndarray], last_ts: list, horizon: int) -> np.ndarray:
    """Recursive forecasts for many origins at once; predictions feed later lags, future actuals are never used."""
    hist = [list(map(float, h[-48:])) for h in histories]
    out = np.zeros((len(hist), horizon))
    for k in range(1, horizon + 1):
        rows = []
        for i, v in enumerate(hist):
            t = pd.Timestamp(last_ts[i]) + np.timedelta64(k, 'h')
            rows.append((t.hour, t.dayofweek, v[-1], v[-24], float(np.mean(v[-24:]))))
        y = np.maximum(model.predict(pd.DataFrame(rows, columns=FEATURES)), 0)
        for i, v in enumerate(hist):
            v.append(float(y[i]))
        out[:, k - 1] = y
    return out


def forecast_meter(df, meter_id, horizon=24):
    """Original API: chronological 80/20 one-step holdout (MAE/RMSE), refit, then a recursive forecast.
    Now runs on a regular hourly grid, so gaps can no longer shift the lag features."""
    if meter_id == TOTAL:
        s = grid(df, TOTAL)
    else:
        out = validate_data(df)
        s = grid(out, meter_id)
    if s.notna().sum() < MIN_MODEL_HOURS:
        raise ValueError('At least 96 readings are required for forecasting')
    sup = features(s)
    split = min(max(1, int(len(sup) * (1 - TEST_SHARE))), len(sup) - 1)
    train, test = sup.iloc[:split], sup.iloc[split:]
    model = make_model().fit(train[FEATURES], train.liters)
    pred = model.predict(test[FEATURES])
    metrics = {'MAE': float(mean_absolute_error(test.liters, pred)),
               'RMSE': float(np.sqrt(mean_squared_error(test.liters, pred))), 'test_rows': len(test),
               'MAE_naive_24h': float(mean_absolute_error(test.liters, test.lag24))}
    model.fit(sup[FEATURES], sup.liters)
    if s.iloc[-24:].isna().any():
        raise ValueError('The last 24 hours contain gaps, so a recursive forecast cannot start.')
    y = _recursive(model, [s.to_numpy()], [s.index[-1]], horizon)[0]
    ts = [s.index[-1] + np.timedelta64(k, 'h') for k in range(1, horizon + 1)]
    return pd.DataFrame({'timestamp': ts, 'forecast_liters': np.round(y, 2)}), metrics


@dataclass
class ForecastEval:
    meter: str
    model: object
    one_step: pd.DataFrame
    day_ahead: pd.DataFrame
    by_horizon: pd.DataFrame
    backtest: pd.DataFrame
    importance: pd.DataFrame
    future: pd.DataFrame
    history: pd.Series
    meta: dict


def evaluate_forecast(df: pd.DataFrame, meter: str, horizon: int = HORIZON, test_share: float = TEST_SHARE,
                      n_estimators: int = 100) -> ForecastEval:
    """Leakage-safe evaluation for one meter (or the network total).

    1. Features are built from past values only. 2. The last ``test_share`` of hours is held out chronologically; the
    model is trained on earlier hours only. 3. One-step-ahead errors use real previous hours (as the original app
    reported). 4. A day-ahead backtest starts at every 23:00 in the holdout and forecasts 24 hours *recursively*,
    exactly like the app's forecast, against two baselines. 5. A final model refit on all hours produces the forecast;
    its band comes from the backtest's empirical 10th–90th percentile errors per horizon."""
    if not 1 <= horizon <= 72:
        raise ValueError('Horizon must be between 1 and 72 hours.')
    s = grid(df, meter)
    if s.notna().sum() < MIN_MODEL_HOURS:
        raise ValueError(f'{meter} has fewer than {MIN_MODEL_HOURS} hourly readings; forecasting needs at least four days.')
    sup = features(s)
    if len(sup) < 72:
        raise ValueError('Not enough gap-free history to build lag features.')
    cut = min(max(24, int(len(sup) * (1 - test_share))), len(sup) - 24)
    train, test = sup.iloc[:cut], sup.iloc[cut:]
    test_model = make_model(n_estimators).fit(train[FEATURES], train.liters)
    p = test_model.predict(test[FEATURES])
    profile = train.groupby('hour').liters.mean().tail(24)
    recent = train[train.index > train.index.max() - pd.Timedelta(days=14)]
    profile14 = recent.groupby('hour').liters.mean()
    one = pd.DataFrame([
        {'method': 'Random Forest, one step ahead (uses the real previous hour)', **_metrics(test.liters, p)},
        {'method': 'Naive: previous hour', **_metrics(test.liters, test.lag1)},
        {'method': 'Seasonal naive: same hour yesterday', **_metrics(test.liters, test.lag24)},
    ])

    y = s.to_numpy()
    idx = s.index
    pos = {t: i for i, t in enumerate(idx)}
    origins = []
    for t in test.index:
        if t.hour != 23:
            continue
        i = pos[t]
        window_ok = i >= 47 and i + horizon < len(y) and not np.isnan(y[i - 47:i + horizon + 1]).any()
        if window_ok and t >= test.index[0] - pd.Timedelta(hours=1):
            origins.append(i)
    rows = []
    if origins:
        pred = _recursive(test_model, [y[:i + 1] for i in origins], [idx[i] for i in origins], horizon)
        for j, i in enumerate(origins):
            for k in range(1, horizon + 1):
                t = idx[i + k]
                rows.append({'origin': idx[i], 'timestamp': t, 'horizon': k, 'actual': y[i + k],
                             'forecast': pred[j, k - 1], 'naive_24h': y[i + k - 24 * int(np.ceil(k / 24))],
                             'profile_14d': profile14.get(t.hour, np.nan)})
    bt = pd.DataFrame(rows)
    if len(bt):
        day = pd.DataFrame([
            {'method': 'Random Forest, recursive day-ahead', **_metrics(bt.actual, bt.forecast)},
            {'method': 'Seasonal naive: same hour on the last observed day', **_metrics(bt.actual, bt.naive_24h)},
            {'method': 'Hour-of-day average (last 14 training days)', **_metrics(bt.actual, bt.profile_14d)},
        ])
        bt['error'] = bt.forecast - bt.actual
        byh = bt.groupby('horizon').agg(
            MAE=('error', lambda e: e.abs().mean()),
            q10=('error', lambda e: e.quantile(.1)), q90=('error', lambda e: e.quantile(.9))).reset_index()
        byh['MAE_naive_24h'] = bt.groupby('horizon').apply(
            lambda g: (g.naive_24h - g.actual).abs().mean(), include_groups=False).to_numpy()
    else:
        day = pd.DataFrame(columns=['method', 'MAE', 'RMSE', 'bias', 'n'])
        byh = pd.DataFrame(columns=['horizon', 'MAE', 'q10', 'q90', 'MAE_naive_24h'])

    imp = (pd.DataFrame({'feature': FEATURES, 'importance': test_model.feature_importances_})
           .assign(label=lambda x: x.feature.map(FEATURE_LABELS)).sort_values('importance', ascending=False)
           .reset_index(drop=True))

    final = make_model(n_estimators).fit(sup[FEATURES], sup.liters)
    if s.iloc[-48:].isna().any():
        future = pd.DataFrame(columns=['timestamp', 'forecast_liters', 'low', 'high'])
    else:
        fy = _recursive(final, [y], [idx[-1]], horizon)[0]
        q = byh.set_index('horizon') if len(byh) else None
        lo = [(q.q90.get(k, np.nan) if q is not None else np.nan) for k in range(1, horizon + 1)]
        hi = [(q.q10.get(k, np.nan) if q is not None else np.nan) for k in range(1, horizon + 1)]
        future = pd.DataFrame({'timestamp': [idx[-1] + np.timedelta64(k, 'h') for k in range(1, horizon + 1)],
                               'forecast_liters': fy,
                               'low': np.maximum(fy - np.asarray(lo, float), 0),     # error = forecast - actual
                               'high': np.maximum(fy - np.asarray(hi, float), 0)})
    meta = {'meter': meter, 'train_hours': int(len(train)), 'test_hours': int(len(test)),
            'train_end': train.index.max(), 'test_start': test.index.min(), 'origins': len(origins),
            'horizon': horizon, 'n_estimators': n_estimators, 'last_observed': idx[-1]}
    return ForecastEval(meter, final, one, day, byh, bt, imp, future, s, meta)


def forecast_summary(ev: ForecastEval) -> dict:
    r = lambda v: round(float(v), 4)
    rec = lambda t: [{k: (r(v) if isinstance(v, (float, np.floating)) else v) for k, v in row.items()}
                     for row in t.to_dict('records')]
    return {'meter': ev.meter, 'one_step': rec(ev.one_step), 'day_ahead': rec(ev.day_ahead),
            'importance': {row.feature: r(row.importance) for row in ev.importance.itertuples()},
            'meta': {k: str(v) for k, v in ev.meta.items()},
            'note': 'Synthetic demonstration data. Not evidence of real-world forecasting accuracy.'}
