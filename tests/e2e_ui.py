"""Browser end-to-end check of the running AquaSense dashboard (Playwright + Chromium).

Not collected by pytest (file name). Usage:
    streamlit run app.py --server.port 8501 &
    python tests/e2e_ui.py http://localhost:8501 [--chrome /path/to/chrome]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.make_messy_sample import build  # noqa: E402

PAGES = [("", "Water Intelligence Command Center", 3), ("consumption", "Consumption Analytics", 4),
         ("anomalies", "Anomaly Detection", 3), ("leaks", "Potential Leak Intelligence", 2),
         ("forecast", "Water Demand Forecasting", 4), ("meters", "Meter Intelligence", 4),
         ("data", "Data Explorer", 1), ("method", "Methodology & Limitations", 0)]


def idle(page, extra=2.0):
    page.wait_for_selector("[data-testid='stAppViewContainer']", timeout=180_000)
    time.sleep(1.5)
    prev = -1
    for _ in range(120):
        busy = page.locator("[data-testid='stSpinner']").count() + page.locator("[data-testid='stStatusWidget']").count()
        n = page.locator(".js-plotly-plot").count()
        if busy == 0 and n == prev:
            break
        prev = n
        time.sleep(1)
    time.sleep(extra)


def kpi(page, i):
    return page.locator("[data-testid='stMain'] .kpi .value").nth(i).inner_text()


def download(page, label) -> bytes:
    with page.expect_download() as dl:
        page.get_by_role("button", name=label).first.click()
    return Path(dl.value.path()).read_bytes()


def sidebar_go(page, title):
    page.locator("[data-testid='stSidebarNav'] a", has_text=title).first.click()
    idle(page)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url", nargs="?", default="http://localhost:8501")
    ap.add_argument("--chrome", default=os.environ.get("CHROME_PATH"))
    a = ap.parse_args()
    base = a.url.rstrip("/")
    fails: list = []
    with sync_playwright() as p:
        launch = {"args": ["--no-sandbox"]}
        if a.chrome:
            launch["executable_path"] = a.chrome
        browser = p.chromium.launch(**launch)
        page = browser.new_page(viewport={"width": 1440, "height": 900}, accept_downloads=True)
        print("Pages:")
        for path, title, min_charts in PAGES:
            page.goto(f"{base}/{path}", wait_until="networkidle")
            idle(page)
            exc = page.locator("[data-testid='stException']").count()
            errs = page.locator("[data-testid='stAlertContentError']").count()
            charts = page.locator(".js-plotly-plot").count()
            ok_title = title in page.locator(".aq-hero h1").first.inner_text()
            print(f"  {path or 'command':<12} exceptions={exc} errors={errs} charts={charts} title={'ok' if ok_title else 'MISSING'}")
            if exc or errs or charts < min_charts or not ok_title:
                fails.append((path, exc, errs, charts, ok_title))

        print("Interactions:")
        page.goto(f"{base}/leaks", wait_until="networkidle")
        idle(page)
        ep = pd.read_csv(io.BytesIO(download(page, "Download episodes (CSV)")))
        higher = ep[ep.priority == "Higher"]
        print(f"  leaks        episodes={len(ep)} higher={len(higher)} meters={sorted(higher.meter_id)}")
        if sorted(higher.meter_id) != ["METER-001", "METER-005"] or not ep.status.str.contains("Potential").all():
            fails.append(("leak episodes", ep.head()))
        body = page.locator("[data-testid='stMain']").inner_text()
        if "2 of 2" not in body:
            fails.append(("synthetic check text",))

        page.goto(f"{base}/forecast", wait_until="networkidle")
        idle(page)
        summ = json.loads(download(page, "Download metrics (JSON)"))
        mae = float(kpi(page, 0).split()[0])
        print(f"  forecast     MAE kpi={mae} json={summ['day_ahead'][0]['MAE']}")
        if abs(mae - summ["day_ahead"][0]["MAE"]) > .01:
            fails.append(("forecast json", mae))
        fc = pd.read_csv(io.BytesIO(download(page, "Download forecast (CSV)")))
        if len(fc) != 24 or not (fc.low <= fc.forecast_liters).all():
            fails.append(("forecast csv", len(fc)))

        page.goto(f"{base}/data", wait_until="networkidle")
        idle(page)
        tmp = Path(tempfile.mkdtemp()) / "messy_meters.csv"
        build().to_csv(tmp, index=False)
        page.locator("[data-testid='stFileUploader'] input[type='file']").set_input_files(str(tmp))
        idle(page, 4)
        body = page.locator("[data-testid='stMain']").inner_text()
        ok = "messy_meters.csv" in body and "WM-DAILY" in body and "converted from cubic metres" in body
        print(f"  upload       accepted and validation report shown={ok}")
        if not ok:
            fails.append(("upload", body[:300]))
        meters = pd.read_csv(io.BytesIO(download(page, "Meter summary")))
        if sorted(meters.meter_id) != ["WM-001", "WM-002", "WM-003"]:
            fails.append(("meter summary", list(meters.meter_id)))
        for title in ["Potential Leak Intelligence", "Water Demand Forecasting", "Water Intelligence Command Center"]:
            sidebar_go(page, title)
            used = page.locator(".pill.ok").count() > 0
            exc = page.locator("[data-testid='stException']").count()
            print(f"  session      {title:<36} uses upload={used} exceptions={exc}")
            if not used or exc:
                fails.append(("session", title))
        sidebar_go(page, "Data Explorer")
        page.get_by_role("button", name="Remove upload").click()
        idle(page)
        back = page.locator(".pill.warn").count() > 0
        print(f"  upload       removed, back to synthetic={back}")
        if not back:
            fails.append(("remove",))
        bad = tmp.with_name("bad.csv")
        pd.DataFrame({"when": ["2026-01-01"], "value": [1]}).to_csv(bad, index=False)
        page.locator("[data-testid='stFileUploader'] input[type='file']").set_input_files(str(bad))
        idle(page, 3)
        err = page.locator("[data-testid='stAlertContentError']").all_inner_texts()
        print(f"  upload       bad file error shown={bool(err)}")
        if not err or "Missing required columns" not in err[0]:
            fails.append(("bad upload", err))
        browser.close()
    print("FAILURES:" if fails else "All browser checks passed.", *fails, sep="\n  ")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
