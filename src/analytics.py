"""AquaSense analytics: illustrative water-meter data, leak signals and forecasts."""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

REQUIRED = {'timestamp', 'meter_id', 'liters'}

def generate_data(days=90, meters=8, seed=42):
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range('2026-01-01', periods=days*24, freq='h')
    rows = []
    for meter in range(meters):
        baseline = rng.uniform(18, 42)
        leak_start = int(rng.integers(len(timestamps)//2, len(timestamps)-48)) if len(timestamps) > 120 else len(timestamps) + 1
        for i, ts in enumerate(timestamps):
            hour = ts.hour
            demand = baseline + 13 * np.sin((hour-7)*np.pi/12)**2 + (8 if 6 <= hour <= 9 or 18 <= hour <= 22 else 0)
            demand += rng.normal(0, 3.2)
            if meter % 4 == 0 and leak_start <= i < leak_start + 40:
                demand += 27
            rows.append((ts, f'METER-{meter+1:03d}', round(max(0, demand), 2)))
    return pd.DataFrame(rows, columns=['timestamp', 'meter_id', 'liters'])

def validate_data(df):
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {", ".join(sorted(missing))}')
    out = df[list(REQUIRED)].copy()
    out['timestamp'] = pd.to_datetime(out['timestamp'], errors='coerce')
    out['liters'] = pd.to_numeric(out['liters'], errors='coerce')
    if out[['timestamp', 'meter_id', 'liters']].isna().any().any():
        raise ValueError('Missing or invalid timestamps, meter IDs or liters')
    if (out['liters'] < 0).any():
        raise ValueError('Liters cannot be negative')
    if out.duplicated(['timestamp', 'meter_id']).any():
        raise ValueError('Duplicate meter/timestamp pairs')
    return out.sort_values(['meter_id', 'timestamp']).reset_index(drop=True)

def add_anomalies(df, contamination=0.04):
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

def leak_signals(df, window=12, threshold=1.35):
    """Heuristic sustained elevated use; not a verified leak diagnosis."""
    out = validate_data(df)
    out['leak_signal'] = False
    for _, part in out.groupby('meter_id'):
        if len(part) < window*3:
            continue
        rolling = part.liters.rolling(window, min_periods=window).mean()
        baseline = part.liters.expanding(min_periods=window).median().shift(window)
        flags = (rolling > baseline * threshold) & (rolling > baseline + 8)
        out.loc[part.index, 'leak_signal'] = flags.fillna(False).to_numpy()
    return out

def forecast_meter(df, meter_id, horizon=24):
    out = validate_data(df)
    part = out[out.meter_id == meter_id].sort_values('timestamp').copy()
    if len(part) < 96:
        raise ValueError('At least 96 readings are required for forecasting')
    part['hour'] = part.timestamp.dt.hour
    part['weekday'] = part.timestamp.dt.dayofweek
    part['lag1'] = part.liters.shift(1)
    part['lag24'] = part.liters.shift(24)
    part['rolling24'] = part.liters.shift(1).rolling(24).mean()
    features = ['hour', 'weekday', 'lag1', 'lag24', 'rolling24']
    supervised = part.dropna(subset=features).copy()
    split = max(1, int(len(supervised)*0.8))
    if split >= len(supervised):
        split = len(supervised)-1
    train, test = supervised.iloc[:split], supervised.iloc[split:]
    model = RandomForestRegressor(n_estimators=100, min_samples_leaf=3, random_state=42, n_jobs=-1)
    model.fit(train[features], train.liters)
    pred = model.predict(test[features])
    metrics = {'MAE': float(mean_absolute_error(test.liters, pred)), 'RMSE': float(np.sqrt(mean_squared_error(test.liters, pred))), 'test_rows': len(test)}
    # Refit on all available observations after holdout evaluation.
    model.fit(supervised[features], supervised.liters)
    history = part.liters.tolist()
    ts = part.timestamp.max()
    future = []
    for _ in range(horizon):
        ts += pd.Timedelta(hours=1)
        row = pd.DataFrame([{'hour': ts.hour, 'weekday': ts.dayofweek, 'lag1': history[-1], 'lag24': history[-24], 'rolling24': np.mean(history[-24:])}])
        value = max(0, float(model.predict(row[features])[0]))
        history.append(value)
        future.append((ts, round(value, 2)))
    return pd.DataFrame(future, columns=['timestamp', 'forecast_liters']), metrics
