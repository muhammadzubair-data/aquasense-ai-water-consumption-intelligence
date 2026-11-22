import pandas as pd
import pytest
from src.analytics import generate_data,validate_data,add_anomalies,leak_signals,forecast_meter

def test_generator_reproducible():
    assert generate_data(days=5,meters=2).equals(generate_data(days=5,meters=2))
def test_generator_size():
    assert len(generate_data(days=5,meters=2))==240
def test_validation_missing():
    with pytest.raises(ValueError): validate_data(pd.DataFrame({'liters':[2]}))
def test_validation_negative():
    with pytest.raises(ValueError): validate_data(pd.DataFrame({'timestamp':['2026-01-01'],'meter_id':['a'],'liters':[-1]}))
def test_validation_duplicate():
    d=generate_data(days=2,meters=1)
    with pytest.raises(ValueError): validate_data(pd.concat([d,d.iloc[:1]]))
def test_anomalies():
    result=add_anomalies(generate_data(days=4,meters=1))
    assert result.anomaly.dtype==bool and len(result)==96
def test_leak_signals():
    result=leak_signals(generate_data(days=4,meters=1))
    assert result.leak_signal.dtype==bool
def test_forecast():
    d=generate_data(days=8,meters=1)
    future,metrics=forecast_meter(d,'METER-001',horizon=6)
    assert len(future)==6 and future.forecast_liters.ge(0).all() and metrics['MAE']>=0
def test_short_forecast():
    with pytest.raises(ValueError): forecast_meter(generate_data(days=2,meters=1),'METER-001')
