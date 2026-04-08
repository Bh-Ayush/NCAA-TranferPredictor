# NCAA Basketball Transfer Portal Success Predictor & Team Ranking Engine

A two-part machine learning system for NCAA basketball analytics: (1) predicting whether transfer portal players will improve at their destination school, and (2) ranking ACC conference teams for the upcoming season using a predictive efficiency model.

Built with **Polars** for data engineering, **DuckDB** for analytical SQL, **XGBoost/LightGBM** for modeling, **Streamlit** for the frontend, and **FastAPI** for serving predictions.

---

## Results Summary

### Part 1 — Transfer Success Prediction Model

| Metric | Fold 1 (Val 2023) | Fold 2 (Val 2024) | Fold 3 (Val 2025) | **Mean** |
|--------|-------------------|-------------------|-------------------|----------|
| PR-AUC | 0.605 | 0.653 | 0.578 | **0.612** |
| ROC-AUC | 0.568 | 0.650 | 0.605 | **0.608** |
| Brier Score | 0.251 | 0.240 | 0.245 | **0.246** |

- **1,149 transfers** across 2022–2025 seasons
- **44 engineered features** spanning individual performance, team context, conference deltas, and interaction terms
- **Temporal cross-validation**: train on seasons ≤N, validate on N+1 (no data leakage)
- Target: binary classification — transfer "succeeds" if post-transfer ORtg ≥ pre-transfer ORtg

### Part 2 — Conference Team Ranking Engine

| Metric | Fold 1 (Val 2022) | Fold 2 (Val 2023) | Fold 3 (Val 2024) | **Mean** |
|--------|-------------------|-------------------|-------------------|----------|
| MAE | 5.90 | 6.19 | 5.29 | **5.80** |
| RMSE | 7.28 | 7.59 | 6.71 | **7.19** |
| R² | 0.429 | 0.467 | 0.548 | **0.481** |

- Trained on **all 191 D1 teams per season**, filtered to **current ACC 18** for output
- Predicts next-season adjusted efficiency margin (adj_o − adj_d)
- Handles conference realignment: Cal, Stanford, SMU have no ACC history — features are conference-agnostic

---

## Project Structure

```
ncaa-predictor/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                        # Parquet files (team, player, transfer, coaching data)
│   └── processed/                  # Engineered feature matrices
├── models/
│   ├── transfer_xgb_final.json     # Trained XGBoost classifier
│   ├── ranking_lgbm_final.txt      # Trained LightGBM regressor
│   ├── acc_rankings.csv            # Predicted ACC 2025-26 rankings
│   ├── project_summary.pdf         # 2-page non-technical summary
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
│   ├── data_generator.py           # Synthetic data (mirrors BartTorvik schemas)
│   ├── scrapers.py                 # Real scrapers for BartTorvik & VerbalCommits
│   ├── feature_engineering.py      # Polars feature pipelines for both models
│   ├── transfer_model.py           # Part 1: XGBoost transfer success classifier
│   ├── ranking_model.py            # Part 2: LightGBM ranking regressor
│   └── duckdb_runner.py            # SQL query executor
├── api/
│   ├── __init__.py
│   └── main.py                     # FastAPI endpoints (5 routes)
└── app/
    └── streamlit_app.py            # Two-tab Streamlit dashboard
```

---

## Methodology

### Transfer Success Model

**Problem framing**: Given a player's pre-transfer stats at their origin school and the characteristics of their destination school, predict whether they will maintain or improve their adjusted offensive rating.

**Feature engineering** (44 features in 6 categories):

1. **Pre-transfer individual performance** — ORtg, usage rate, BPM, eFG%, TS%, assist/TO rates, rebound rates, stocks (blocks + steals), FT rate, PORPAG
2. **Origin team context** — Adjusted offensive/defensive efficiency, Barthag, tempo
3. **Destination team context** — Same metrics for the receiving school
4. **Contextual deltas** — Destination minus origin quality (Barthag delta, adj_o delta, tempo delta), conference jump direction (power ↔ mid-major)
5. **Player profile** — Recruiting star rating, class year, height
6. **Interaction terms** — Usage × quality delta, MPG × conference jump, BPM × offensive delta

**Evaluation**: Temporal cross-validation with PR-AUC as the primary metric (appropriate for the ~52/48 class balance). Calibration curve confirms predicted probabilities are well-calibrated.

### Team Ranking Engine

**Problem framing**: Given a team's current-season performance metrics, returning production, portal activity, and coaching stability, predict their next-season adjusted efficiency margin.

**Key design decision**: Model is trained on all ~191 D1 teams per season to maximize learning signal, then filtered to the **current 2025-26 ACC membership** (18 teams including Cal, Stanford, SMU) for presentation. This handles realignment gracefully — features are conference-agnostic efficiency metrics, not conference-specific historical rankings.

**Features** (17):
- Prior season: adj_o, adj_d, adj_t, Barthag, efficiency margin, WAB, win%, SOS
- Roster continuity: returning production %, incoming transfers count, transfer composite rating, average transfer quality
- Coaching: tenure years, stability bucket
- Context: power conference indicator, returning production × quality interaction

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
| API (Phase 3) | **FastAPI** | Prediction endpoint serving |
| Dashboard (Phase 3) | **Streamlit** | Interactive frontend |

---

## Data Sources

**Current implementation** uses a synthetic data generator (`src/data_generator.py`) that mirrors exact BartTorvik schemas with realistic distributions. To use real data:

1. **BartTorvik** (barttorvik.com) — Player and team advanced stats, 2021–2025
2. **VerbalCommits** (verbalcommits.com) — Historical transfer portal entries
3. **247Sports** — Composite recruiting star ratings

Run `python src/scrapers.py --all` locally (requires unrestricted internet) to pull real data. The pipeline automatically uses whatever parquet files are in `data/raw/`.

---

## Quickstart

```bash
# Clone and setup
git clone https://github.com/Bh-Ayush/ncaa-transfer-predictor.git
cd ncaa-transfer-predictor
python -m venv venv && source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Generate synthetic data (or run scrapers for real data)
python -c "from src.data_generator import generate_all_data; [df.write_parquet(f'data/raw/{k}.parquet') for k, df in generate_all_data().items()]"

# Run both models
python src/transfer_model.py    # Part 1: Transfer Success Predictor
python src/ranking_model.py     # Part 2: ACC Team Ranking Engine

# Run SQL queries
python src/duckdb_runner.py

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
- **Tab 1 — Transfer Predictor**: Interactive form with all 44 features, real-time prediction with probability gauge, confidence tier, and context summary
- **Tab 2 — ACC Rankings**: Horizontal bar chart, sortable table, current-vs-predicted scatter plot, embedded model evaluation plots

---

## Key Findings

### Transfer Success Patterns

- **Mid-major → Power conference** transfers have the highest success rate (58.9%), suggesting these players were undervalued at their origin
- **Power → Mid-major** transfers show the lowest success rate (36.6%) — players stepping down in competition still struggle
- **Juniors** show the highest success rate among class years, likely due to experience + remaining eligibility
- Transfers to **Tier 1 destinations** (top 25% by Barthag) succeed 62% of the time, compared to 43.5% for Tier 4

### Ranking Model Insights

- **Prior season efficiency margin** and **Barthag** are the strongest predictors of next-season performance (SHAP)
- **Returning production %** is the most important non-efficiency feature — teams that retain their core improve
- **Coaching tenure** shows diminishing returns past ~6 years
- The model improves with more training data (R² increases from 0.43 → 0.55 across folds)

---

## Author

**Ayush Bhardwaj** — [GitHub](https://github.com/Bh-Ayush) | [Portfolio](https://ayush-hb.com)

B.S. Data Science, Arizona State University | M.S. Business Analytics (Aug 2026)
