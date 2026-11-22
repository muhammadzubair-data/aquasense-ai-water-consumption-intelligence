# Model card: AquaSense AI

| | |
|:--|:--|
| **Components** | Context-aware Isolation Forest (unusual readings) · hour-aware sustained-excess episodes with night-flow priority (potential leak indicators) · Random Forest hourly demand forecaster |
| **Version** | 2.0.0 (`aquasense/core.py`) |
| **Owner** | ZubairLabs, Muhammad Zubair |
| **Intended use** | Demonstrating water-consumption analytics and honest evaluation on hourly meter data |
| **Not intended for** | Confirming leaks, dispatching crews, billing adjustments, shut-offs, safety or regulatory decisions, or any customer-facing judgement |
| **Data** | Synthetic by default (90 days × 8 meters, seed 42), or an uploaded CSV held in session memory |

## Inputs
Hourly `timestamp`, `meter_id` and `liters`. No customer, pressure, weather or network data.

## Methods
- **Typical value:** median of the same hour on earlier days of the same type (10 weekdays / 4 weekend days, at least 3), using past data only.
- **Unusual readings:** `IsolationForest(n_estimators=200, contamination=rate, random_state=42)` per meter, on the reading, its gap from typical, and the 3-hour mean gap.
- **Potential leak indicators:**
  - signal when the 12-hour mean is at least 1.35 × typical and at least 8 L/h above it
  - signal runs of 6 hours or more form an episode
  - **higher priority** when the episode lasts at least 12 hours and night flow (02:00–04:59) is raised in at least 2 hours
- **Forecaster:**
  - `RandomForestRegressor(n_estimators=100, min_samples_leaf=3, random_state=42)` on hour, weekday, previous hour, same hour yesterday and trailing 24-hour mean
  - chronological 80/20 holdout
  - recursive day-ahead backtest from every 23:00 in the holdout
  - band = 10th–90th percentile of backtest errors per hours-ahead

## Results on synthetic data (seed 42, realistic profile, pinned deployment versions)

| Check | Result |
|:--|--:|
| Higher-priority episodes: injected leaks found / non-matching | 2 of 2 / 0 (also on seeds 1, 7, 11 and the original profile) |
| All episodes: non-matching (all lower priority) | 6 |
| Mean delay from leak start | 3.5 h |
| Unusual readings at 1%: burst hours / leak hours flagged | 49 of 49 / 28 of 80 |
| Day-ahead MAE, network total: RF vs best baseline | 15.50 vs 27.73 L/h |
| Day-ahead MAE per meter: RF | 2.58–5.46 L/h, lower than both baselines on all 8 meters |
| Original profile | Hour-of-day average as good or better on 8 of 9 series |

## Ethical and operational considerations
- **Not a diagnosis:** a potential indicator is not a leak. Legitimate use (guests, filling a pool, irrigation) can trigger one, and slowly growing leaks may be missed.
- **Wrong in both directions:** false alarms cost site visits and customer trust, while missed leaks cost water. Thresholds should be tuned with confirmed outcomes before any real use.
- **Assumed rate:** the contamination rate is an assumption; the forest always flags that share.
- **Privacy:** household meter data can reveal routines and occupancy. Uploads stay in the browser session, and the public demo should not receive customer data.

## Caveats
- Synthetic leaks are clean steps, and synthetic noise is Gaussian.
- There are no weather, holiday or occupancy effects.
- Meters are treated independently.
- Recursive errors compound, and the forecast band is not calibrated.
