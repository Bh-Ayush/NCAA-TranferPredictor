"""
FastAPI prediction endpoints for both models.

Endpoints:
  POST /predict/transfer    — Predict transfer success probability
  POST /predict/rankings    — Get ACC team rankings
  GET  /health              — Health check
  GET  /model/transfer/info — Transfer model metadata
  GET  /model/ranking/info  — Ranking model metadata

Run:
  uvicorn api.main:app --reload --port 8000
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import xgboost as xgb
import lightgbm as lgb

from src.feature_engineering import get_transfer_feature_columns, get_ranking_feature_columns


# ─── App Setup ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="NCAA Basketball Transfer Portal Predictor & Ranking Engine",
    description=(
        "Two-model API: (1) Predict whether a transfer portal player will "
        "improve at their destination school, and (2) Generate ACC conference "
        "team rankings for the upcoming season."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = Path("models")
DATA_DIR = Path("data")


# ─── Model Loading ───────────────────────────────────────────────────────────

def load_transfer_model() -> xgb.XGBClassifier:
    path = MODELS_DIR / "transfer_xgb_final.json"
    if not path.exists():
        raise FileNotFoundError(f"Transfer model not found at {path}. Run src/transfer_model.py first.")
    model = xgb.XGBClassifier()
    model.load_model(str(path))
    return model


def load_ranking_model() -> lgb.Booster:
    path = MODELS_DIR / "ranking_lgbm_final.txt"
    if not path.exists():
        raise FileNotFoundError(f"Ranking model not found at {path}. Run src/ranking_model.py first.")
    return lgb.Booster(model_file=str(path))


def load_model_metrics(name: str) -> dict:
    path = MODELS_DIR / f"{name}_model_metrics.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


# Lazy-load models on first request
_transfer_model: Optional[xgb.XGBClassifier] = None
_ranking_model: Optional[lgb.Booster] = None


def get_transfer_model() -> xgb.XGBClassifier:
    global _transfer_model
    if _transfer_model is None:
        _transfer_model = load_transfer_model()
    return _transfer_model


def get_ranking_model() -> lgb.Booster:
    global _ranking_model
    if _ranking_model is None:
        _ranking_model = load_ranking_model()
    return _ranking_model


# ─── Request/Response Schemas ────────────────────────────────────────────────

class TransferInput(BaseModel):
    """Input for a single transfer prediction."""
    # Pre-transfer individual stats
    pre_ortg: float = Field(..., description="Pre-transfer offensive rating (per 100 poss)")
    pre_usg: float = Field(..., description="Pre-transfer usage rate (%)")
    pre_efg: float = Field(..., description="Pre-transfer effective FG%")
    pre_ts_pct: float = Field(..., description="Pre-transfer true shooting %")
    pre_ast_pct: float = Field(..., description="Pre-transfer assist rate")
    pre_to_pct: float = Field(..., description="Pre-transfer turnover rate")
    pre_orb_pct: float = Field(..., description="Pre-transfer offensive rebound rate")
    pre_drb_pct: float = Field(..., description="Pre-transfer defensive rebound rate")
    pre_blk_pct: float = Field(..., description="Pre-transfer block rate")
    pre_stl_pct: float = Field(..., description="Pre-transfer steal rate")
    pre_ftr: float = Field(..., description="Pre-transfer free throw rate")
    pre_porpag: float = Field(..., description="Pre-transfer points over replacement")
    pre_bpm: float = Field(..., description="Pre-transfer box plus/minus")
    pre_obpm: float = Field(..., description="Pre-transfer offensive BPM")
    pre_dbpm: float = Field(..., description="Pre-transfer defensive BPM")
    pre_mpg: float = Field(..., description="Pre-transfer minutes per game")
    pre_g: int = Field(..., description="Pre-transfer games played")

    # Player profile
    recruiting_stars: int = Field(..., ge=0, le=5, description="247 composite star rating")
    class_year_ord: int = Field(..., ge=1, le=4, description="1=Fr, 2=So, 3=Jr, 4=Sr")
    height_in: int = Field(..., description="Height in inches")

    # Origin team context
    origin_adj_o: float = Field(..., description="Origin team adjusted offensive efficiency")
    origin_adj_d: float = Field(..., description="Origin team adjusted defensive efficiency")
    origin_adj_t: float = Field(..., description="Origin team adjusted tempo")
    origin_barthag: float = Field(..., description="Origin team Barthag rating")
    origin_is_power: int = Field(..., ge=0, le=1, description="1 if origin is Power conference")

    # Destination team context
    dest_adj_o: float = Field(..., description="Destination team adjusted offensive efficiency")
    dest_adj_d: float = Field(..., description="Destination team adjusted defensive efficiency")
    dest_adj_t: float = Field(..., description="Destination team adjusted tempo")
    dest_barthag: float = Field(..., description="Destination team Barthag rating")
    dest_is_power: int = Field(..., ge=0, le=1, description="1 if destination is Power conference")

    class Config:
        json_schema_extra = {
            "example": {
                "pre_ortg": 108.5, "pre_usg": 22.3, "pre_efg": 0.52,
                "pre_ts_pct": 0.56, "pre_ast_pct": 14.2, "pre_to_pct": 16.8,
                "pre_orb_pct": 4.5, "pre_drb_pct": 12.3, "pre_blk_pct": 2.1,
                "pre_stl_pct": 1.8, "pre_ftr": 0.35, "pre_porpag": 3.2,
                "pre_bpm": 2.5, "pre_obpm": 1.8, "pre_dbpm": 0.7,
                "pre_mpg": 28.5, "pre_g": 30,
                "recruiting_stars": 3, "class_year_ord": 2, "height_in": 77,
                "origin_adj_o": 105.2, "origin_adj_d": 98.4, "origin_adj_t": 68.0,
                "origin_barthag": 0.72, "origin_is_power": 0,
                "dest_adj_o": 112.8, "dest_adj_d": 94.1, "dest_adj_t": 70.5,
                "dest_barthag": 0.88, "dest_is_power": 1,
            }
        }


class TransferPrediction(BaseModel):
    success_probability: float
    predicted_class: str
    confidence: str
    feature_contributions: dict


class RankingsResponse(BaseModel):
    season: str
    teams: list[dict]
    model_info: dict


# ─── Feature Computation ────────────────────────────────────────────────────

def compute_derived_features(inp: TransferInput) -> dict:
    """Compute all engineered features from raw input."""
    base = inp.model_dump()

    # Deltas
    base["delta_adj_o"] = inp.dest_adj_o - inp.origin_adj_o
    base["delta_adj_d"] = inp.dest_adj_d - inp.origin_adj_d
    base["delta_barthag"] = inp.dest_barthag - inp.origin_barthag
    base["delta_tempo"] = inp.dest_adj_t - inp.origin_adj_t
    base["conf_jump_direction"] = inp.dest_is_power - inp.origin_is_power

    # Engineered
    base["pre_ortg_usg_product"] = inp.pre_ortg * inp.pre_usg / 100
    base["pre_minutes_share"] = inp.pre_mpg / 40
    base["pre_ast_to_ratio"] = inp.pre_ast_pct / max(inp.pre_to_pct, 1)
    base["pre_shooting_composite"] = inp.pre_efg + 0.1 * inp.pre_ftr
    base["pre_total_reb_pct"] = inp.pre_orb_pct + inp.pre_drb_pct
    base["pre_defensive_stocks"] = inp.pre_blk_pct + inp.pre_stl_pct

    # Interactions
    base["usg_x_quality_delta"] = inp.pre_usg * base["delta_barthag"]
    base["mpg_x_conf_jump"] = inp.pre_mpg * float(base["conf_jump_direction"])
    base["bpm_x_off_delta"] = inp.pre_bpm * base["delta_adj_o"]

    return base


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {"status": "healthy", "models_loaded": {
        "transfer": _transfer_model is not None,
        "ranking": _ranking_model is not None,
    }}


@app.get("/model/transfer/info")
def transfer_model_info():
    metrics = load_model_metrics("transfer")
    return {
        "model_type": "XGBoost Classifier",
        "target": "Transfer success (post-transfer ORtg >= pre-transfer ORtg)",
        "n_features": len(get_transfer_feature_columns()),
        "features": get_transfer_feature_columns(),
        "metrics": metrics,
    }


@app.get("/model/ranking/info")
def ranking_model_info():
    metrics = load_model_metrics("ranking")
    return {
        "model_type": "LightGBM Regressor",
        "target": "Next-season adjusted efficiency margin",
        "n_features": len(get_ranking_feature_columns()),
        "features": get_ranking_feature_columns(),
        "metrics": metrics,
    }


@app.post("/predict/transfer", response_model=TransferPrediction)
def predict_transfer(inp: TransferInput):
    """Predict whether a transfer will succeed."""
    try:
        model = get_transfer_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Build feature vector
    features = compute_derived_features(inp)
    feature_cols = get_transfer_feature_columns()

    X = pd.DataFrame([{col: features[col] for col in feature_cols}])

    # Predict
    prob = float(model.predict_proba(X)[0, 1])
    predicted_class = "Success" if prob >= 0.5 else "Decline"

    # Confidence tier
    if prob >= 0.7 or prob <= 0.3:
        confidence = "High"
    elif prob >= 0.6 or prob <= 0.4:
        confidence = "Medium"
    else:
        confidence = "Low"

    # Top feature contributions (simple importance-weighted direction)
    importances = model.feature_importances_
    top_indices = np.argsort(importances)[-5:][::-1]
    contributions = {}
    for idx in top_indices:
        col = feature_cols[idx]
        val = features[col]
        contributions[col] = {
            "value": round(float(val), 3),
            "importance": round(float(importances[idx]), 4),
        }

    return TransferPrediction(
        success_probability=round(prob, 4),
        predicted_class=predicted_class,
        confidence=confidence,
        feature_contributions=contributions,
    )


@app.get("/predict/rankings", response_model=RankingsResponse)
def predict_rankings():
    """Get predicted ACC team rankings for the upcoming season."""
    try:
        model = get_ranking_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Load ACC rankings if pre-computed
    csv_path = MODELS_DIR / "acc_rankings.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        teams = []
        for _, row in df.iterrows():
            teams.append({
                "rank": int(row["predicted_rank"]),
                "team": row["team"],
                "predicted_eff_margin": round(float(row["predicted_eff_margin"]), 2),
                "current_eff_margin": round(float(row["current_eff_margin"]), 2),
                "predicted_change": round(float(row["predicted_change"]), 2),
                "rank_change": int(row["rank_change"]),
                "returning_production_pct": round(float(row["returning_production_pct"]), 3),
                "coaching_tenure_years": int(row["coaching_tenure_years"]),
            })

        metrics = load_model_metrics("ranking")
        return RankingsResponse(
            season="2025-26",
            teams=teams,
            model_info={
                "type": "LightGBM Regressor",
                "avg_mae": metrics.get("avg_mae"),
                "avg_r2": metrics.get("avg_r2"),
                "trained_on": "All D1 teams, filtered to current ACC 18",
            },
        )

    raise HTTPException(
        status_code=503,
        detail="Rankings not yet generated. Run src/ranking_model.py first.",
    )


# ─── Run ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
