# AquaSense AI: portfolio and announcement material

Live app: LIVE_URL
Code: https://github.com/muhammadzubair-data/aquasense-ai-water-consumption-intelligence

All figures come from synthetic demonstration data. They show the method, not real-world detection or forecasting accuracy.

---

## 1. Technical case study

**AquaSense AI: telling a leak apart from a busy afternoon**

**Context.** Smart water meters report every hour, but most dashboards either flood teams with alerts or call any
high reading a "leak". I wanted a tool that separates three questions:
- Is this reading unusual?
- Has use stayed above normal?
- Does the night-time pattern point to a continuous leak?

I also wanted every claim checked against known ground truth.

**What I built.** An eight-module Streamlit app on a tested analytics package:
- consumption analytics with hourly profiles, a weekly heatmap and peak hours
- context-aware anomaly detection
- potential-leak episodes ranked by night flow
- a Random Forest demand forecaster with honest baselines
- meter comparisons
- a forgiving CSV importer
- a methodology page that shows live synthetic results

**Engineering decisions**
- **Keep the old, measure the new.** The original generator, anomaly model, leak rule and forecast metric are still there and reproduced exactly (tested). Every change is scored against them on the same data.
- **A realistic profile with traps.** The original data had night use equal to daytime use. I added a residential profile with low night flow and short legitimate bursts, such as irrigation or filling a tank, so that "unusual" and "leak" can be told apart.
- **Hour-aware baselines.** Each reading is compared with the same hour on earlier days of the same type, using past data only. A test changes future values and confirms past baselines don't move.
- **Night flow as the tiebreaker.** On the synthetic data, higher-priority episodes (12+ hours above normal *and* raised 02:00–05:00 flow) found 2 of 2 injected leaks with 0 false alarms. They did the same on 8 seed and profile combinations. The 6 lower-priority episodes are still shown, but clearly ranked lower.
- **Unusual is not leak.** At a 1% flag rate, anomaly detection caught all 49 legitimate burst hours but only 28 of 80 leak hours. That result is why the modules are separate.
- **Score what users see.** v1 reported one-step-ahead errors while showing multi-hour forecasts. The new backtest forecasts 24 hours recursively from every 23:00 in a chronological holdout. A test proves it never sees future values.
  - Random Forest MAE: 2.6–5.5 L/h per meter and 15.5 L/h for the network, beating both baselines everywhere on the realistic data.
  - On the original data a simple hour-of-day average wins, and the app says so.
- **Messy data welcome.** The importer handles m³ and gallons, time-zone offsets (keeping local clock time), duplicates, 15-minute readings, short and long gaps, and daily-only meters. It reports every repair.

**Quality.** 72 automated tests (the original 9 untouched), a Playwright browser test and GitHub Actions on Python 3.11 and 3.12.

**Limits I'm upfront about.** The data is synthetic and there are no confirmed leak labels or pressure data. Indicators
are signals for a site check, not diagnoses.

---

## 2. LinkedIn announcement

A water meter that reads high at 3 pm tells you very little. One that reads high at 3 am tells you a lot.

I built AquaSense AI, a smart-water analytics app that tries to tell the difference between a leak and someone watering the garden.

What it does:
- Hourly and daily consumption analytics, peak hours and meter comparisons
- Flags readings that are unusual for that hour of the day (Isolation Forest)
- Finds sustained above-normal use and ranks it higher when night flow rises too, the classic sign of a continuous leak
- Forecasts the next 6–72 hours with a Random Forest and checks it against simple baselines
- Cleans messy meter exports: m³, gallons, time zones, 15-minute data, gaps

What the synthetic tests showed:
→ Higher-priority indicators found both injected leaks, with zero false alarms, across 8 test runs
→ Anomaly detection caught every legitimate burst but only a third of leak hours. Unusual is not the same as a leak.
→ The forecast beat both baselines on every meter, but on the original "too tidy" data a plain hourly average won, and the app says so

All results are from synthetic data. These are signals for a site check, not confirmed leaks.

Live demo: LIVE_URL
Code: https://github.com/muhammadzubair-data/aquasense-ai-water-consumption-intelligence

#WaterAnalytics #SmartUtilities #DataScience #MachineLearning #AnomalyDetection #Forecasting #Streamlit

---

## 3. Upwork portfolio entry (plain text)

Title:
AquaSense AI: Smart Water Consumption, Leak Indicator and Demand Forecasting Dashboard (Python, Streamlit, ML)

Description:
A water analytics web app for hourly smart-meter data. It shows where and when water is used, which readings are
unusual, which meters show sustained above-normal use worth a site check, and what demand to expect next.

What it does:
- 8 dashboard modules: command center, consumption analytics, anomaly detection, potential leak intelligence, demand forecasting, meter intelligence, data explorer and methodology
- Hourly and daily trends, peak hours, a weekly usage heatmap and meter-by-meter comparison
- Isolation Forest anomaly detection that compares each reading with what is normal for that hour
- Potential leak episodes ranked by night-time flow, with estimated extra litres
- Random Forest forecasts for 6 to 72 hours, with an error band and comparison against simple baselines
- CSV upload that handles m3 or gallons, time zones, 15-minute readings, duplicates and gaps, with a clear validation report
- Light and dark themes, CSV and JSON exports

Results on synthetic test data:
- Higher-priority leak indicators found 2 of 2 injected leaks with 0 false alarms across 8 test runs
- Forecast error of 2.6 to 5.5 litres per hour per meter, lower than both simple baselines

Honest limits: synthetic data, no confirmed leak labels or pressure data; indicators are signals to check, not confirmed leaks.

Tools: Python, pandas, scikit-learn, Plotly, Streamlit, pytest, Playwright, GitHub Actions

Live demo: LIVE_URL
Code: https://github.com/muhammadzubair-data/aquasense-ai-water-consumption-intelligence

---

## 4. Six LinkedIn carousel concepts

1. **"3 pm vs 3 am"**
   - Two meter charts with the same high reading: one in the afternoon (normal), one at night (suspicious).
   - Close: "Context is the feature."
2. **"Unusual ≠ leak"**
   - Show the anomaly check: 49 of 49 burst hours flagged, only 28 of 80 leak hours.
   - Explain why anomaly detection and leak indicators need to be separate tools.
3. **"Night flow breaks the tie"**
   - The episode table, with lower-priority daytime rises next to the two higher-priority episodes that had raised night flow.
   - Result: 2 of 2 leaks, 0 false alarms.
4. **"Score what users see"**
   - One-step error vs a 24-hour recursive forecast.
   - Explain why the backtest must feed its own predictions back in, and show the test that proves it never peeks.
5. **"When the baseline wins"**
   - On the original data, a plain hour-of-day average beat the Random Forest.
   - Lesson: always publish the baseline.
6. **"Messy meter data in, clean answers out"**
   - Before: 15-minute readings in m³ with a time-zone offset, duplicates and gaps.
   - After: the validation report, with 6,025 rows becoming 1,500 hourly readings and every fix counted.

---

## 5. ZubairLabs website project description

**AquaSense AI: Smart Water Consumption Intelligence**

An eight-module dashboard for hourly water-meter data. It covers:
- consumption trends and meter comparisons
- anomaly detection that knows what is normal for each hour
- potential-leak episodes ranked by night-time flow
- Random Forest demand forecasts checked against simple baselines

It cleans messy meter exports (units, time zones, 15-minute data, gaps) and reports every fix. It was tested against
known synthetic leaks: higher-priority indicators found both, with no false alarms. Built with Python, scikit-learn,
Plotly and Streamlit, with automated tests and CI. Demonstrated on synthetic data, with indicators presented as
signals to check, not confirmed leaks.

[Live demo](LIVE_URL) · [Source code](https://github.com/muhammadzubair-data/aquasense-ai-water-consumption-intelligence)
