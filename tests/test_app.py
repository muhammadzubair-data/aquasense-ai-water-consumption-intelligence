"""Streamlit AppTest checks: every page renders, uploads are used everywhere, and controls take effect."""
from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from aquasense.core import prepare
from scripts.make_messy_sample import build

APP = str(Path(__file__).resolve().parents[1] / "app.py")
PAGES = ["", "consumption", "anomalies", "leaks", "forecast", "meters", "data", "method"]


def run(page="", **state):
    at = AppTest.from_file(APP, default_timeout=240)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    if page:
        at._page_hash = next(h for h, i in at._registered_pages.items() if i["url_pathname"] == page)
        at.run()
    return at


def text(at):
    return " ".join(m.value for m in at.markdown)


def upload_state():
    clean, rep = prepare(build())
    return dict(upload_df=clean, upload_report=rep, upload_name="meters.csv", upload_key="k1", use_upload=True)


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = run(page)
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error
    assert "Synthetic demonstration data" in text(at)


@pytest.mark.parametrize("page", ["", "anomalies", "leaks", "forecast", "meters", "data"])
def test_upload_used_everywhere(page):
    at = run(page, **upload_state())
    assert not at.exception, [e.value for e in at.exception]
    assert "Your uploaded file" in text(at)
    assert "WM-" in " ".join(str(s.options) for s in at.sidebar.selectbox)


def test_original_profile_and_method_switch():
    at = run("anomalies")
    at.sidebar.selectbox[0].set_value("Original (project v1)").run()
    assert not at.exception and "original profile" in text(at)
    at.radio[0].set_value("Original v1 (raw reading, hour, weekday)").run()
    assert not at.exception


def test_leak_settings_change_results():
    at = run("leaks")
    before = text(at)
    at.slider[1].set_value(1.8).run()           # above-normal ratio
    assert not at.exception and text(at) != before


def test_forecast_series_and_horizon():
    at = run("forecast")
    at.selectbox[0].set_value("ALL METERS").run()
    at.select_slider[0].set_value(48).run()
    assert not at.exception
    assert "Next 48 h total" in text(at)


def test_original_import_path():
    from src.analytics import add_anomalies, forecast_meter, generate_data, leak_signals, validate_data  # noqa: F401
    assert len(generate_data(days=2, meters=1)) == 48
