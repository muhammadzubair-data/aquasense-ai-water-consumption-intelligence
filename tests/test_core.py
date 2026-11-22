"""Extended tests: generator fidelity, upload preparation, analytics, anomalies, potential leaks and forecasting."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error

from aquasense.core import (FEATURES, MIN_MODEL_HOURS, REQUIRED, TOTAL, _parse_times, add_anomalies, daily,
                            detect_anomalies, episode_scores, evaluate_forecast, expected_same_hour, features,
                            flag_scores, forecast_meter, forecast_summary, generate_data, grid, hourly_profile, kpis,
                            leak_signals, leak_windows, meter_summary, night_flow, peak_periods, potential_leaks,
                            prepare, validate_data, weekday_hour)


def original_generate(days=90, meters=8, seed=42):
    """Verbatim copy of the original row-by-row generator, used to prove the data did not change."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range('2026-01-01', periods=days * 24, freq='h')
    rows = []
    for meter in range(meters):
        baseline = rng.uniform(18, 42)
        leak_start = int(rng.integers(len(timestamps) // 2, len(timestamps) - 48)) if len(timestamps) > 120 else len(timestamps) + 1
        for i, ts in enumerate(timestamps):
            hour = ts.hour
            demand = baseline + 13 * np.sin((hour - 7) * np.pi / 12) ** 2 + (8 if 6 <= hour <= 9 or 18 <= hour <= 22 else 0)
            demand += rng.normal(0, 3.2)
            if meter % 4 == 0 and leak_start <= i < leak_start + 40:
                demand += 27
            rows.append((ts, f'METER-{meter + 1:03d}', round(max(0, demand), 2)))
    return pd.DataFrame(rows, columns=['timestamp', 'meter_id', 'liters'])


@pytest.fixture(scope='module')
def real():
    return generate_data(truth=True, profile='realistic')


@pytest.fixture(scope='module')
def orig():
    return generate_data(truth=True)


# ------------------------------------------------------------------ generator
@pytest.mark.parametrize('args', [dict(), dict(days=5, meters=2), dict(days=30, meters=12, seed=7)])
def test_generator_matches_original(args):
    assert generate_data(**args).equals(original_generate(**args))


def test_committed_sample_csv_is_the_generator():
    csv = pd.read_csv('data/synthetic_water_meters.csv', parse_dates=['timestamp'])
    assert csv.equals(generate_data(days=14, meters=4))


def test_truth_columns(orig, real):
    w = leak_windows(orig)
    assert list(w.meter_id) == ['METER-001', 'METER-005']
    assert orig.groupby('meter_id').injected_leak.sum().tolist() == [40, 0, 0, 0, 40, 0, 0, 0]
    assert real.injected_burst.sum() > 0 and not (real.injected_burst & real.injected_leak).any()
    assert real[orig.columns.drop('liters')].equals(orig[orig.columns.drop('liters')])   # same meters, times, leaks


def test_realistic_profile_shape(real):
    prof = real.groupby(real.timestamp.dt.hour).liters.mean()
    assert prof.loc[[2, 3, 4]].mean() < .3 * prof.mean()        # low night flow
    assert prof.idxmax() in (7, 8, 18, 19, 20)
    with pytest.raises(ValueError):
        generate_data(5, 1, profile='nope')


# ------------------------------------------------------------------ upload preparation
def messy():
    from scripts.make_messy_sample import build
    return build()


def test_prepare_messy_sample():
    c, r = prepare(messy())
    assert list(c.columns) == REQUIRED
    assert (r.dropped_timestamp, r.dropped_meter, r.dropped_liters, r.dropped_negative) == (1, 1, 2, 1)
    assert (r.exact_duplicates, r.conflicting_duplicates) == (10, 2)
    assert r.resampled_meters == 3 and r.excluded_meters == ['WM-DAILY']
    assert (r.interpolated_hours, r.missing_hours) == (2, 12)
    assert r.renamed['Reading Time'] == 'timestamp' and 'm3' in r.unit_note.lower() + 'm3'
    assert r.ignored_columns == ['Customer Note']
    assert not c.duplicated(['meter_id', 'timestamp']).any()
    assert c.groupby('meter_id').timestamp.apply(lambda t: t.is_monotonic_increasing).all()
    # local clock kept: the +05:00 file starts at local midnight, and night is the quietest time
    assert str(c.timestamp.min()) == '2026-01-01 00:00:00'
    prof = c.groupby(c.timestamp.dt.hour).liters.mean()
    assert prof.loc[[2, 3, 4]].mean() < prof.mean() * .4


def test_prepare_volume_conserved_on_resample():
    c, _ = prepare(messy())
    hourly = generate_data(21, 3, seed=5, profile='realistic')
    # WM-002 had no injected damage except shared duplicates removed → totals match the source within rounding
    src = hourly[hourly.meter_id == 'METER-002'].liters.sum()
    got = c[c.meter_id == 'WM-002'].liters.sum()
    assert got == pytest.approx(src, rel=1e-3)


def test_prepare_units_and_aliases():
    base = generate_data(3, 1)
    gal = base.rename(columns={'timestamp': 'datetime', 'meter_id': 'Meter', 'liters': 'gallons'})
    gal['gallons'] = base.liters / 3.785411784
    c, r = prepare(gal)
    assert c.liters.sum() == pytest.approx(base.liters.sum(), rel=1e-4)
    assert 'gallons' in r.unit_note


@pytest.mark.parametrize('bad,match', [
    (pd.DataFrame({'liters': [1.0]}), 'Missing required columns'),
    (pd.DataFrame({'timestamp': ['x'] * 3, 'meter_id': ['a'] * 3, 'liters': [1, 2, 3]}), 'No valid readings'),
    (generate_data(1, 1), 'fewer than 48'),
    (pd.DataFrame({'timestamp': pd.date_range('2026-01-01', periods=30, freq='D'), 'meter_id': 'a', 'liters': 1.0}), 'hourly'),
])
def test_prepare_errors(bad, match):
    with pytest.raises(ValueError, match=match):
        prepare(bad)


def test_prepare_limits():
    d = generate_data(3, 2)
    with pytest.raises(ValueError, match='limit'):
        prepare(d, max_rows=10)
    with pytest.raises(ValueError, match='meters found'):
        prepare(d, max_meters=1)


def test_parse_times_variants():
    t, n = _parse_times(pd.Series(['2026-02-01T13:00:00+05:00', '2026-02-01T14:00:00+05:00']))
    assert list(t.dt.hour) == [13, 14] and 'local' in n
    t, n = _parse_times(pd.Series(['2026-03-01 01:00:00+01:00', '2026-03-01 02:00:00+02:00']))
    assert t.nunique() == 1 and 'UTC' in n
    t, _ = _parse_times(pd.Series(['2026-02-01 13:00', 'junk']))
    assert t.isna().sum() == 1


def test_validate_data_strict_and_ordered():
    v = validate_data(generate_data(2, 2))
    assert list(v.columns) == REQUIRED
    with pytest.raises(ValueError):
        validate_data(pd.DataFrame({'timestamp': ['2026-01-01'], 'meter_id': ['a'], 'liters': [None]}))


# ------------------------------------------------------------------ analytics
def test_kpis_and_aggregates(real):
    d = real[REQUIRED]
    k = kpis(d)
    assert k['total_liters'] == pytest.approx(d.liters.sum())
    assert daily(d).liters.sum() == pytest.approx(d.liters.sum())
    assert daily(d, by_meter=True).groupby('meter_id').liters.sum().sum() == pytest.approx(d.liters.sum())
    ms = meter_summary(d)
    assert ms.share.sum() == pytest.approx(1)
    assert ms.total_liters.sum() == pytest.approx(d.liters.sum())
    assert hourly_profile(d)['mean'].between(0, 200).all()
    assert weekday_hour(d).shape == (7, 24)
    pk = peak_periods(d, 5)
    assert pk.liters.is_monotonic_decreasing and len(pk) == 5


def test_night_flow_flags_leak_nights(real):
    nf = night_flow(real[REQUIRED])
    top = nf.sort_values('ratio', ascending=False).head(2)
    w = leak_windows(real)
    for r in top.itertuples():
        win = w[w.meter_id == r.meter_id].iloc[0]
        assert win.start.normalize() <= r.date <= win.end.normalize()


def test_grid_inserts_gaps():
    d = generate_data(5, 1).drop(index=[30, 31, 32])
    s = grid(d, 'METER-001')
    assert len(s) == 120 and s.isna().sum() == 3 and s.index.freqstr == 'h'
    tot = grid(generate_data(5, 2), TOTAL)
    assert tot.iloc[0] == pytest.approx(generate_data(5, 2).groupby('timestamp').liters.sum().iloc[0])


# ------------------------------------------------------------------ anomalies
def test_expected_uses_past_only():
    s = grid(generate_data(30, 1), 'METER-001')
    e1 = expected_same_hour(s)
    s2 = s.copy()
    s2.iloc[-24:] = 9999                                    # changing the future must not change earlier baselines
    e2 = expected_same_hour(s2)
    assert e1.iloc[:-24].equals(e2.iloc[:-24])


def test_detect_anomalies_rate_and_columns(real):
    a = detect_anomalies(real[REQUIRED], contamination=0.01)
    assert {'expected', 'excess', 'anomaly', 'anomaly_score', 'direction'} <= set(a.columns)
    rate = a.groupby('meter_id').anomaly.mean()
    assert ((rate - .01).abs() < .005).all()


def test_anomalies_catch_bursts_more_than_leaks(real):
    a = detect_anomalies(real[REQUIRED], contamination=0.01)
    bursts = flag_scores(a, real.assign(injected_leak=real.injected_burst), 'anomaly')
    leaks = flag_scores(a, real, 'anomaly')
    assert bursts['recall'] > .9 and leaks['recall'] < .6      # unusual ≠ leak


def test_context_anomalies_beat_original_precision(real):
    any_event = real.assign(injected_leak=real.injected_leak | real.injected_burst)
    for rate in (.005, .01, .02):
        new = flag_scores(detect_anomalies(real[REQUIRED], rate), any_event, 'anomaly')
        old = flag_scores(add_anomalies(real[REQUIRED], rate), any_event, 'anomaly')
        assert new['precision'] >= old['precision']
    assert new['precision'] > 1.3 * old['precision']            # the gap widens as more readings are flagged


# ------------------------------------------------------------------ potential leaks
@pytest.mark.parametrize('profile', ['original', 'realistic'])
@pytest.mark.parametrize('seed', [42, 1, 7, 11])
def test_higher_priority_finds_every_leak_without_false_alarms(profile, seed):
    d = generate_data(truth=True, profile=profile, seed=seed)
    _, ep = potential_leaks(d[REQUIRED])
    s = episode_scores(ep[ep.priority == 'Higher'], d)
    assert s['leaks_found'] == s['injected_leaks'] == 2 and s['false_episodes'] == 0
    assert 0 <= s['mean_delay_hours'] <= 12


def test_leak_episode_fields_and_bursts(real):
    h, ep = potential_leaks(real[REQUIRED])
    assert set(ep.priority) <= {'Higher', 'Lower'} and (ep.hours >= 6).all()
    assert ep.status.str.contains('Potential').all()
    burst_hours = h.merge(real, on=['timestamp', 'meter_id']).query('injected_burst')
    assert not burst_hours.merge(ep[ep.priority == 'Higher'], on='episode').shape[0]


def test_stricter_settings_reduce_episodes(real):
    n_default = len(potential_leaks(real[REQUIRED])[1])
    strict = potential_leaks(real[REQUIRED], ratio=1.8, min_excess=15, min_hours=12)[1]
    s = episode_scores(strict, real)
    assert len(strict) < n_default and s['false_episodes'] == 0
    assert s['leaks_found'] == 1                                 # the trade-off: stricter settings can miss a leak


def test_original_leak_signal_kept(orig):
    o = leak_signals(orig[REQUIRED])
    assert o.leak_signal.dtype == bool and flag_scores(o, orig, 'leak_signal')['recall'] > .8


# ------------------------------------------------------------------ forecasting
def test_features_past_only():
    s = grid(generate_data(10, 1), 'METER-001')
    f = features(s)
    t = f.index[50]
    assert f.loc[t, 'lag1'] == s[t - pd.Timedelta(hours=1)]
    assert f.loc[t, 'lag24'] == s[t - pd.Timedelta(hours=24)]
    assert f.loc[t, 'rolling24'] == pytest.approx(s[t - pd.Timedelta(hours=24):t - pd.Timedelta(hours=1)].mean())


def test_forecast_meter_original_metric_unchanged():
    f, m = forecast_meter(generate_data(), 'METER-002', 24)
    assert m['MAE'] == pytest.approx(2.780685533406487) and m['test_rows'] == 428
    assert len(f) == 24 and f.forecast_liters.ge(0).all()


def test_forecast_meter_handles_gaps_on_grid():
    d = generate_data(10, 1).drop(index=range(100, 103))
    f, m = forecast_meter(d, 'METER-001', 6)
    assert len(f) == 6 and m['MAE'] >= 0


@pytest.fixture(scope='module')
def ev_real(real):
    return evaluate_forecast(real[REQUIRED], 'METER-002')


def test_evaluation_chronological_and_recomputable(real, ev_real):
    ev = ev_real
    assert ev.meta['train_end'] < ev.meta['test_start']
    bt = ev.backtest
    assert (bt.origin >= ev.meta['train_end']).all()
    assert ev.day_ahead.iloc[0].MAE == pytest.approx(mean_absolute_error(bt.actual, bt.forecast))
    assert len(ev.by_horizon) == 24 and ev.meta['origins'] == len(bt) // 24


def test_backtest_never_sees_future(real, ev_real):
    """Changing every value after an origin must not change that origin's forecast."""
    d = real[REQUIRED].copy()
    origin = ev_real.backtest.origin.iloc[0]
    mask = (d.meter_id == 'METER-002') & (d.timestamp > origin)
    d.loc[mask, 'liters'] = d.loc[mask, 'liters'] * 3
    ev2 = evaluate_forecast(d, 'METER-002')
    a = ev_real.backtest[ev_real.backtest.origin == origin].forecast.to_numpy()
    b = ev2.backtest[ev2.backtest.origin == origin].forecast.to_numpy()
    assert np.allclose(a, b)


def test_random_forest_beats_baselines_on_realistic(ev_real):
    day = ev_real.day_ahead.set_index('method').MAE
    assert day.iloc[0] < day.iloc[1:].min()
    assert ev_real.importance.iloc[0].feature == 'lag24'


def test_simple_profile_can_win_on_original():
    ev = evaluate_forecast(generate_data(), 'METER-002')
    day = ev.day_ahead.set_index('method').MAE
    assert day['Hour-of-day average (last 14 training days)'] < day['Random Forest, recursive day-ahead']


def test_future_band_and_total(real):
    ev = evaluate_forecast(real[REQUIRED], TOTAL, horizon=48)
    f = ev.future
    assert len(f) == 48 and (f.low <= f.forecast_liters + 1e-9).all() and (f.high >= f.forecast_liters - 1e-9).all()
    assert json.loads(json.dumps(forecast_summary(ev), default=str))['meter'] == TOTAL


def test_forecast_errors():
    with pytest.raises(ValueError, match='fewer than'):
        evaluate_forecast(generate_data(3, 1), 'METER-001')
    with pytest.raises(ValueError, match='Horizon'):
        evaluate_forecast(generate_data(10, 1), 'METER-001', horizon=200)


def test_evaluation_deterministic(real, ev_real):
    assert evaluate_forecast(real[REQUIRED], 'METER-002').day_ahead.equals(ev_real.day_ahead)


def test_constants():
    assert FEATURES == ['hour', 'weekday', 'lag1', 'lag24', 'rolling24'] and MIN_MODEL_HOURS == 96
