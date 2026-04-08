# NCAA Basketball Transfer Portal Predictor & Team Ranking Engine

An end-to-end analytics system built to answer two questions that coaching staffs and front offices face every offseason:

1. **Which transfer portal players are most likely to improve at our program?**
2. **How will our conference stack up next season based on roster movement and returning talent?**

Trained on **8 seasons of real BartTorvik data** (2018-2025) covering 39,263 player-seasons and 2,851 team-seasons across all of Division I basketball. The system scrapes live data, engineers 44 features per transfer and 17 features per team, and produces predictions through an interactive dashboard and API.

---

## Why This Matters

The transfer portal has fundamentally changed college basketball. Programs are rebuilding rosters year-over-year, and the difference between a good and bad portal class can swing a season by 5-10 wins. But most transfer evaluation is still based on gut feel and highlight film.

This system quantifies transfer fit by combining **player efficiency data** (ORtg, usage, BPM, shooting splits) with **team context** (adjusted offensive/defensive efficiency, Barthag, tempo) and **conference dynamics** (power vs. mid-major jumps, quality deltas). It answers questions like:

- Will a mid-major guard putting up 18 PPG actually produce at a Power 5 program, or is he a system player?
- Is a bench player at a blue blood worth targeting if he transfers to a program where he'll start?
- How does a team's projected efficiency margin change after adding 3 portal players and losing its top scorer?

---

## Results

### Transfer Success Predictor -- 82% Precision, 73% Accuracy

The model correctly identifies whether a transfer will improve or decline at their new school **73.4% of the time**, with particularly strong performance on successful transfers (79% recall). This means when the model flags a portal player as a good fit, it's right about 4 out of 5 times.

| What We Measured | Score | What It Means |
|-----------------|-------|---------------|
| **PR-AUC** | **0.820** | The model ranks likely-successful transfers above likely-unsuccessful ones 82% of the time |
| **ROC-AUC** | **0.788** | Strong ability to separate good fits from bad fits across all confidence levels |
| **Accuracy** | **73.4%** | Out of 4,178 transfers evaluated, the model got the right call on ~3,066 |
| **Brier Score** | **0.185** | When the model says "70% chance of success," that closely matches the actual success rate |

**Trained on 4,560 real transfers** derived from BartTorvik player data. Validated using walk-forward temporal cross-validation: the model only ever predicts future seasons using past data, so there is no information leakage.

<details>
<summary>Detailed fold-by-fold results</summary>

| Metric | Fold 1 (Val 2021) | Fold 2 (Val 2022) | Fold 3 (Val 2023) | Fold 4 (Val 2024) | Fold 5 (Val 2025) | **Mean** |
|--------|-------------------|-------------------|-------------------|-------------------|-------------------|----------|
| PR-AUC | 0.728 | 0.838 | 0.848 | 0.833 | 0.855 | **0.820** |
| ROC-AUC | 0.746 | 0.821 | 0.811 | 0.757 | 0.805 | **0.788** |
| Brier Score | 0.219 | 0.172 | 0.173 | 0.186 | 0.174 | **0.185** |

Each fold trains on all prior seasons and validates on the next, simulating how the model would perform in real-time use.
</details>

### ACC Team Ranking Engine -- Explains 71% of Next-Season Performance

The ranking model predicts next-season adjusted efficiency margin for every D1 team, then filters to the current ACC. It captures **71% of the variance** in how teams actually perform the following year -- meaning roster continuity, portal activity, and coaching stability are genuinely predictive, not just noise.

| What We Measured | Score | What It Means |
|-----------------|-------|---------------|
| **R-squared** | **0.708** | The model explains 71% of why some teams improve and others decline season-over-season |
| **MAE** | **5.15** | Predictions are off by about 5 points of efficiency margin on average (on a scale where the gap between a tournament team and a bubble team is ~10 points) |
| **RMSE** | **6.42** | Typical prediction error; larger misses are penalized more heavily |

**Trained on 2,474 team-seasons** spanning all D1 programs. The model learns from the full landscape of college basketball, then generates ranked projections for the ACC's 18 current members (including Cal, Stanford, and SMU post-realignment).

<details>
<summary>Detailed fold-by-fold results</summary>

| Metric | Fold 1 (Val 2020) | Fold 2 (Val 2021) | Fold 3 (Val 2022) | Fold 4 (Val 2023) | Fold 5 (Val 2024) | **Mean** |
|--------|-------------------|-------------------|-------------------|-------------------|-------------------|----------|
| MAE | 4.81 | 5.26 | 4.99 | 5.31 | 5.41 | **5.15** |
| RMSE | 5.93 | 6.50 | 6.24 | 6.60 | 6.84 | **6.42** |
| R-squared | 0.751 | 0.679 | 0.698 | 0.704 | 0.710 | **0.708** |

</details>

---

## How It Works

### Data Collection

The system scrapes real NCAA basketball data from **BartTorvik** -- one of the most trusted public sources for college basketball advanced stats, widely used by analysts, media, and programs.

| Data Source | Records | What It Contains |
|-------------|---------|------------------|
| Team stats (JSON) | 2,851 team-seasons | Adjusted O/D efficiency, Barthag, tempo, SOS, WAB for ~350 D1 teams/year |
| Player stats (CSV) | 39,263 player-seasons | ORtg, usage, eFG%, TS%, assist/TO rates, rebound rates, BPM, PORPAG for ~5,000 players/year |
| Transfers (derived) | 4,560 matched transfers | Identified by tracking player IDs across seasons -- pre and post stats captured automatically |
| Coaching tenure | 2,851 records | Derived from team data across seasons |
| Returning production | 2,466 records | Computed from player minutes returning to the same team |

Transfers are **not scraped from a third-party portal tracker**. Instead, the system identifies transfers directly from BartTorvik player data: when the same player ID appears on a different team the following season, we know they transferred, and we automatically have both their before and after performance stats. This is more reliable than name-matching across external sources.

### Feature Engineering -- What the Model Actually Sees

**Transfer model (44 features):**

The model does not just look at box scores. It evaluates each transfer through the lens of how a player's skill set fits the destination program:

- **Player ability**: ORtg, usage rate, BPM, shooting efficiency (eFG%, TS%), assist-to-turnover ratio, rebound rates, defensive stocks (blocks + steals), free throw rate, PORPAG (points over replacement)
- **Where they're coming from**: Origin team's adjusted offensive/defensive efficiency, Barthag (win probability vs. average D1 team), tempo
- **Where they're going**: Same metrics for the destination team
- **The gap between the two**: Quality deltas (is the player stepping up or stepping down in competition?), conference jump direction (mid-major to Power 5, or vice versa)
- **Player profile**: Class year, height, recruiting background
- **Interaction effects**: How a player's usage rate interacts with the quality gap, how their minutes relate to the conference jump, how their BPM connects to the offensive system change

**Ranking model (17 features):**

- **Prior season efficiency**: KenPom-style adjusted O/D, Barthag, efficiency margin, WAB, win%, strength of schedule
- **Roster continuity**: What percentage of last year's minutes are returning, how many transfers are incoming, and how good those transfers were
- **Coaching stability**: Tenure and whether the program has a new, establishing, or established coach
- **Conference context**: Power conference indicator, interaction between returning talent and prior quality

### Validation Approach

Both models use **temporal cross-validation** -- the same approach you'd use if deploying this in practice. The model never sees future data when making predictions:

- Fold 1 trains on 2018-2020 and predicts 2021
- Fold 2 trains on 2018-2021 and predicts 2022
- ...and so on through 2025

This prevents the "peeking at the answer key" problem that inflates accuracy in many sports models. Every number reported here reflects true out-of-sample performance.

---

## Analytical Insights

### Transfer Portal Patterns (from 4,560 real transfers)

- **Mid-major to Power conference** transfers show the highest success rate -- these players are often undervalued relative to the system they're entering
- **Power to mid-major** transfers show the lowest success rate -- stepping down in competition does not guarantee better stats
- **Juniors** transfer most successfully, likely because they combine experience with remaining eligibility
- Transfers to **top-quartile programs** (by Barthag) succeed 62% of the time, compared to 44% for bottom-quartile destinations -- system and coaching matter

### What Drives Next-Season Team Performance

According to SHAP feature importance analysis:
- **Prior season efficiency margin and Barthag** are the strongest predictors -- good teams tend to stay good
- **Returning production %** is the most important roster factor -- teams that keep their core intact project better
- **Coaching tenure** matters up to ~6 years, then shows diminishing returns
- **Portal transfer quality** is predictive but secondary to returning your own players

---

## SQL Analytics (DuckDB)

10 analytical queries in `sql/queries.sql` that demonstrate the kind of ad-hoc analysis this data supports:

| # | Question Answered | Techniques Used |
|---|-------------------|-----------------|
| 1 | How does transfer success vary by conference direction? | CTE, CASE WHEN |
| 2 | Which programs attract the most portal talent? | RANK() window, HAVING |
| 3 | How do pre/post stats differ by class year? | AVG OVER() window |
| 4 | Who are the most efficient players relative to their conference? | RANK() PARTITION BY |
| 5 | How has a team's efficiency trended over 3 years? | LAG, ROWS BETWEEN |
| 6 | What does a team's roster look like after portal activity? | LEFT JOIN, COALESCE |
| 7 | Do transfers to better programs actually succeed more? | NTILE() |
| 8 | Who were the most impactful transfers each season? | ROW_NUMBER, QUALIFY |
| 9 | Does coaching stability affect portal usage and results? | Multi-table JOIN |
| 10 | How has transfer volume grown year-over-year? | LAG, percentage change |

---

## Interactive Dashboard & API

### Streamlit Dashboard

A two-tab interactive interface for exploring predictions without writing code:

- **Transfer Predictor tab**: Enter any player's stats and team context, get a real-time success probability with a confidence gauge and context breakdown
- **ACC Rankings tab**: Visual rankings with predicted efficiency margins, current-vs-predicted comparisons, and embedded model evaluation plots

### REST API (FastAPI)

For programmatic access and integration into existing workflows:

| Method | Endpoint | What It Returns |
|--------|----------|-----------------|
| GET | `/health` | System status |
| GET | `/model/transfer/info` | Model metadata, feature list, accuracy metrics |
| GET | `/model/ranking/info` | Same for ranking model |
| POST | `/predict/transfer` | Success probability for a specific transfer scenario |
| GET | `/predict/rankings` | Full ACC rankings table with predicted efficiency margins |

---

## Tech Stack

| Layer | Tools | Why |
|-------|-------|-----|
| Data engineering | **Python, Polars** | Fast columnar operations on 39K+ player records |
| Analytical queries | **DuckDB, SQL** | Ad-hoc analysis directly on parquet files |
| Transfer prediction | **XGBoost** | Gradient-boosted trees optimized for PR-AUC |
| Team ranking | **LightGBM** | Efficient regression on 2,400+ team-seasons |
| Explainability | **SHAP** | Feature importance and individual prediction explanations |
| Visualization | **matplotlib, seaborn, Plotly** | Evaluation plots and interactive charts |
| API | **FastAPI** | Production-ready prediction serving |
| Dashboard | **Streamlit** | Interactive frontend for non-technical users |

---

## Quickstart

```bash
# Clone and setup
git clone https://github.com/Bh-Ayush/NCAA-TranferPredictor.git
cd NCAA-TranferPredictor
python -m venv venv && source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Scrape real data from BartTorvik (~2 min with rate limiting)
python src/scrapers.py --all

# Train both models and generate evaluation plots
python -m src.transfer_model    # Transfer Success Predictor
python -m src.ranking_model     # ACC Team Ranking Engine

# Run analytical SQL queries
python -m src.duckdb_runner

# Launch the API server
uvicorn api.main:app --reload --port 8000

# Launch the Streamlit dashboard (separate terminal)
streamlit run app/streamlit_app.py
```

---

## Project Structure

```
ncaa-predictor/
├── src/
│   ├── scrapers.py                 # BartTorvik data collection + transfer derivation
│   ├── feature_engineering.py      # 44-feature transfer pipeline, 17-feature ranking pipeline
│   ├── transfer_model.py           # XGBoost classifier with temporal CV
│   ├── ranking_model.py            # LightGBM regressor with ACC rankings output
│   ├── data_generator.py           # Synthetic data fallback for testing
│   └── duckdb_runner.py            # SQL query executor
├── api/
│   └── main.py                     # FastAPI with 5 prediction endpoints
├── app/
│   └── streamlit_app.py            # Two-tab interactive dashboard
├── sql/
│   └── queries.sql                 # 10 analytical queries
├── notebooks/
│   ├── 01_transfer_eda.ipynb       # Transfer portal exploratory analysis
│   └── 02_ranking_eda.ipynb        # Team ranking exploratory analysis
├── generate_report.py              # Stakeholder PDF report generator
├── requirements.txt
└── README.md
```

Data, models, and plots are gitignored (regenerated by running the pipeline).

---

## Author

**Ayush Bhardwaj** -- [GitHub](https://github.com/Bh-Ayush) | [Portfolio](https://ayush-hb.com)

B.S. Data Science, Arizona State University | M.S. Business Analytics (Aug 2026)
