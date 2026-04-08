# NCAA Basketball Transfer Portal Success Predictor & Team Ranking Engine

A two-part machine learning system for NCAA basketball analytics: (1) predicting whether transfer portal players will improve at their destination school, and (2) ranking ACC conference teams for the upcoming season using a predictive efficiency model.

Built with **Polars** for data engineering, **DuckDB** for analytical SQL, **XGBoost/LightGBM** for modeling, **Streamlit** for the frontend, and **FastAPI** for serving predictions. Trained on **real BartTorvik data** spanning 2018-2025 (39,263 player-seasons, 2,851 team-seasons).

---

## Results Summary

### Part 1 -- Transfer Success Prediction Model

| Metric | Fold 1 (Val 2021) | Fold 2 (Val 2022) | Fold 3 (Val 2023) | Fold 4 (Val 2024) | Fold 5 (Val 2025) | **Mean** |
|--------|-------------------|-------------------|-------------------|-------------------|-------------------|----------|
| PR-AUC | 0.728 | 0.838 | 0.848 | 0.833 | 0.855 | **0.820** |
| ROC-AUC | 0.746 | 0.821 | 0.811 | 0.757 | 0.805 | **0.788** |
| Brier Score | 0.219 | 0.172 | 0.173 | 0.186 | 0.174 | **0.185** |

- **4,560 real transfers** derived from BartTorvik player data across 2018-2025 seasons
- **44 engineered features** spanning individual performance, team context, conference deltas, and interaction terms
- **73.4% overall accuracy** with 79% recall on successful transfers
- **Temporal cross-validation**: train on seasons <=N, validate on N+1 (no data leakage, min 2 training seasons)
- Target: binary classification -- transfer "succeeds" if post-transfer ORtg >= pre-transfer ORtg

### Part 2 -- Conference Team Ranking Engine

| Metric | Fold 1 (Val 2020) | Fold 2 (Val 2021) | Fold 3 (Val 2022) | Fold 4 (Val 2023) | Fold 5 (Val 2024) | **Mean** |
|--------|-------------------|-------------------|-------------------|-------------------|-------------------|----------|
| MAE | 4.81 | 5.26 | 4.99 | 5.31 | 5.41 | **5.15** |
| RMSE | 5.93 | 6.50 | 6.24 | 6.60 | 6.84 | **6.42** |
| R-squared | 0.751 | 0.679 | 0.698 | 0.704 | 0.710 | **0.708** |

- Trained on **all 350+ D1 teams per season** (2,474 team-seasons total), filtered to **current ACC 18** for output
- Predicts next-season adjusted efficiency margin (adj_o - adj_d)
- **R-squared of 0.71** means the model explains 71% of the variance in next-season team performance
- Handles conference realignment: Cal, Stanford, SMU have no ACC history -- features are conference-agnostic

---

## Project Structure

```
ncaa-predictor/
├── README.md
├── requirements.txt
├── .gitignore
├── generate_report.py              # Stakeholder PDF report generator
├── data/
│   ├── raw/                        # Parquet files (scraped from BartTorvik)
│   └── processed/                  # Engineered feature matrices
├── models/
│   ├── transfer_xgb_final.json     # Trained XGBoost classifier
│   ├── ranking_lgbm_final.txt      # Trained LightGBM regressor
│   ├── acc_rankings.csv            # Predicted ACC 2025-26 rankings
│   ├── transfer_model_metrics.json
│   └── ranking_model_metrics.json
├── notebooks/
│   ├── 01_transfer_eda.ipynb       # Transfer portal EDA (pre-rendered outputs)
│   └── 02_ranking_eda.ipynb        # Team ranking EDA (pre-rendered outputs)
├── plots/
│   ├── transfer_pr_curve.png       # Precision-Recall curve (temporal CV)
│   ├── transfer_calibration.png    # Calibration + prediction distribution
│   ├── transfer_confusion.png      # Confusion matrix
│   ├── transfer_shap.png           # SHAP feature importance
│   ├── acc_rankings.png            # ACC predicted rankings bar chart
│   ├── ranking_actual_vs_pred.png
│   ├── ranking_residuals.png
│   └── ranking_shap.png
├── sql/
│   └── queries.sql                 # 10 analytical queries (DuckDB)
├── src/
│   ├── __init__.py
│   ├── data_generator.py           # Synthetic data fallback (mirrors BartTorvik schemas)
│   ├── scrapers.py                 # BartTorvik scrapers + transfer derivation
│   ├── feature_engineering.py      # Polars feature pipelines for both models
│   ├── transfer_model.py           # Part 1: XGBoost transfer success classifier
│   ├── ranking_model.py            # Part 2: LightGBM ranking regressor
│   └── duckdb_runner.py            # SQL query executor
├── api/
│   └── main.py                     # FastAPI endpoints (5 routes)
└── app/
    └── streamlit_app.py            # Two-tab Streamlit dashboard
```

---

## Data Pipeline

The system scrapes real NCAA basketball data from **BartTorvik** (barttorvik.com), a leading source for college basketball advanced stats:

1. **Team stats** -- scraped from BartTorvik's JSON endpoint (`{year}_team_results.json`), covering ~350 D1 teams per season with adjusted offensive/defensive efficiency, Barthag, tempo, SOS, and WAB
2. **Player stats** -- scraped from BartTorvik's CSV endpoint (`getadvstats.php`), covering ~5,000 players per season with ORtg, usage, shooting splits, BPM, and more
3. **Transfers** -- derived by tracking player IDs across consecutive seasons; when a player appears on a different team, we capture both their pre-transfer and post-transfer stats automatically (4,560 transfers across 2018-2025)
4. **Coaching tenure** -- derived from team stats (coach column when available, default otherwise)
5. **Returning production** -- computed from player data by measuring what fraction of a team's prior-season minutes return

```
python src/scrapers.py --all    # Scrapes everything (~2 min with rate limiting)
```

---

## Methodology

### Transfer Success Model

**Problem framing**: Given a player's pre-transfer stats at their origin school and the characteristics of their destination school, predict whether they will maintain or improve their adjusted offensive rating.

**Feature engineering** (44 features in 6 categories):

1. **Pre-transfer individual performance** -- ORtg, usage rate, BPM, eFG%, TS%, assist/TO rates, rebound rates, stocks (blocks + steals), FT rate, PORPAG
2. **Origin team context** -- Adjusted offensive/defensive efficiency, Barthag, tempo
3. **Destination team context** -- Same metrics for the receiving school
4. **Contextual deltas** -- Destination minus origin quality (Barthag delta, adj_o delta, tempo delta), conference jump direction (power vs mid-major)
5. **Player profile** -- Recruiting star rating, class year, height
6. **Interaction terms** -- Usage x quality delta, MPG x conference jump, BPM x offensive delta

**Evaluation**: Temporal cross-validation (5 folds, min 2 training seasons per fold) with PR-AUC as the primary metric. Calibration curve confirms predicted probabilities are well-calibrated.

### Team Ranking Engine

**Problem framing**: Given a team's current-season performance metrics, returning production, portal activity, and coaching stability, predict their next-season adjusted efficiency margin.

**Key design decision**: Model is trained on all ~350 D1 teams per season to maximize learning signal, then filtered to the **current 2025-26 ACC membership** (18 teams including Cal, Stanford, SMU) for presentation. This handles realignment gracefully -- features are conference-agnostic efficiency metrics, not conference-specific historical rankings.

**Features** (17):
- Prior season: adj_o, adj_d, adj_t, Barthag, efficiency margin, WAB, win%, SOS
- Roster continuity: returning production %, incoming transfers count, transfer composite rating, average transfer quality
- Coaching: tenure years, stability bucket
- Context: power conference indicator, returning production x quality interaction

---

## SQL Queries (DuckDB)

The `sql/queries.sql` file contains 10 analytical queries demonstrating:

| # | Query | SQL Techniques |
|---|-------|----------------|
| 1 | Transfer success rate by conference jump direction | CTE, CASE WHEN |
| 2 | Top transfer destinations by incoming talent | RANK() window, HAVING |
| 3 | Pre/post performance by class year | AVG OVER() window |
| 4 | Conference-adjusted player efficiency rankings | RANK() PARTITION BY, CTE join |
| 5 | Rolling 3-year team efficiency trend | LAG, ROWS BETWEEN |
| 6 | Team roster composition via portal | LEFT JOIN, COALESCE |
| 7 | Transfer outcomes by destination quality tier | NTILE() |
| 8 | Most impactful transfers per season (top 10) | ROW_NUMBER, QUALIFY |
| 9 | Coaching tenure vs performance & portal activity | Multi-table JOIN |
| 10 | Season-over-season transfer volume trends | LAG, percentage change |

---

## Tech Stack

| Component | Tool | Purpose |
|-----------|------|---------|
| Data ingestion & cleaning | **Polars** | Fast columnar operations, lazy evaluation |
| Modeling pipeline | **pandas** | scikit-learn/XGBoost compatibility at model boundary |
| Analytical queries | **DuckDB** | SQL on parquet files, zero-copy integration |
| Transfer classifier | **XGBoost** | Gradient-boosted trees, PR-AUC optimization |
| Ranking regressor | **LightGBM** | Efficient gradient boosting for regression |
| Explainability | **SHAP** | TreeExplainer for feature importance |
| Visualization | **matplotlib/seaborn** | Evaluation plots |
| API | **FastAPI** | Prediction endpoint serving |
| Dashboard | **Streamlit** | Interactive frontend |

---

## Quickstart

```bash
# Clone and setup
git clone https://github.com/Bh-Ayush/NCAA-TranferPredictor.git
cd NCAA-TranferPredictor
python -m venv venv && source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Scrape real data from BartTorvik (~2 min)
python src/scrapers.py --all

# Run both models
python -m src.transfer_model    # Part 1: Transfer Success Predictor
python -m src.ranking_model     # Part 2: ACC Team Ranking Engine

# Run SQL queries
python -m src.duckdb_runner

# Launch API server
uvicorn api.main:app --reload --port 8000

# Launch Streamlit dashboard (in a separate terminal)
streamlit run app/streamlit_app.py
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check + model load status |
| GET | `/model/transfer/info` | Transfer model metadata and metrics |
| GET | `/model/ranking/info` | Ranking model metadata and metrics |
| POST | `/predict/transfer` | Predict transfer success (accepts player stats + team context) |
| GET | `/predict/rankings` | Get predicted ACC 2025-26 rankings |

### Streamlit Dashboard

Two-tab interface:
- **Tab 1 -- Transfer Predictor**: Interactive form with all 44 features, real-time prediction with probability gauge, confidence tier, and context summary
- **Tab 2 -- ACC Rankings**: Horizontal bar chart, sortable table, current-vs-predicted scatter plot, embedded model evaluation plots

---

## Author

**Ayush Bhardwaj** -- [GitHub](https://github.com/Bh-Ayush) | [Portfolio](https://ayush-hb.com)

B.S. Data Science, Arizona State University | M.S. Business Analytics (Aug 2026)
