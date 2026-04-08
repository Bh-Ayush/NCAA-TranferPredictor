"""
Part 2 — Conference Team Ranking System

LightGBM regressor predicting next-season adjusted efficiency margin
for all D1 teams. Output filtered to current ACC (2025-26 membership).

Trained on all D1 teams to maximize sample size, presented for ACC only.
Compared against prior-season rankings as a sanity check.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import shap

from src.feature_engineering import (
    build_ranking_features,
    get_ranking_feature_columns,
    temporal_split_ranking,
)


MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

CURRENT_ACC_TEAMS = [
    "Florida St.", "Clemson", "North Carolina", "Duke", "Virginia",
    "Louisville", "Pittsburgh", "NC State", "Wake Forest", "Syracuse",
    "Georgia Tech", "Boston College", "Notre Dame", "Miami FL",
    "Virginia Tech", "California", "Stanford", "SMU",
]


# ─── Temporal Cross-Validation ───────────────────────────────────────────────

def temporal_cv_ranking(features: pl.DataFrame,
                        feature_cols: list[str],
                        target_col: str = "next_adj_eff_margin",
                        ) -> dict:
    """
    Walk-forward temporal CV for the ranking model.

    Folds: Train through season N, validate on N+1.
    Each fold trains on ALL D1 teams (not just ACC).
    """
    seasons = sorted(features["year"].unique().to_list())
    # Start from index 1 so fold 1 trains on at least 2 seasons
    start_idx = min(1, len(seasons) - 2)
    n_folds = len(seasons) - 1 - start_idx
    print(f"Available feature seasons: {seasons}")
    print(f"Temporal CV folds: {n_folds} (requiring >=2 training seasons)")

    fold_results = []

    for i in range(start_idx, len(seasons) - 1):
        train_through = seasons[i]

        X_train, y_train, X_val, y_val, meta_val = temporal_split_ranking(
            features, train_through, feature_cols, target_col
        )

        if len(X_val) < 20:
            continue

        val_season = train_through + 1
        fold_num = i - start_idx + 1
        print(f"\n--- Fold {fold_num}: Train <={train_through}, Val={val_season} ---")
        print(f"  Train: {len(X_train)} team-seasons")
        print(f"  Val:   {len(X_val)} team-seasons")

        # Train LightGBM regressor
        model = lgb.LGBMRegressor(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=10,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )

        # Predictions
        y_pred = model.predict(X_val)

        # Metrics
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)

        print(f"  MAE:  {mae:.3f}")
        print(f"  RMSE: {rmse:.3f}")
        print(f"  R²:   {r2:.4f}")

        # ACC-specific results
        acc_mask = meta_val["team"].isin(CURRENT_ACC_TEAMS)
        if acc_mask.sum() > 0:
            acc_mae = mean_absolute_error(y_val[acc_mask], y_pred[acc_mask])
            print(f"  ACC MAE: {acc_mae:.3f} ({acc_mask.sum()} teams)")
        else:
            acc_mae = None

        fold_results.append({
            "fold": fold_num,
            "train_through": train_through,
            "val_season": val_season,
            "n_train": len(X_train),
            "n_val": len(X_val),
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "acc_mae": acc_mae,
            "model": model,
            "y_val": y_val,
            "y_pred": y_pred,
            "X_val": X_val,
            "meta_val": meta_val,
        })

    avg_mae = np.mean([f["mae"] for f in fold_results])
    avg_rmse = np.mean([f["rmse"] for f in fold_results])
    avg_r2 = np.mean([f["r2"] for f in fold_results])

    print(f"\n=== Aggregated Temporal CV Results ===")
    print(f"  Mean MAE:  {avg_mae:.3f}")
    print(f"  Mean RMSE: {avg_rmse:.3f}")
    print(f"  Mean R²:   {avg_r2:.4f}")

    return {
        "folds": fold_results,
        "avg_mae": avg_mae,
        "avg_rmse": avg_rmse,
        "avg_r2": avg_r2,
    }


# ─── Final Model & ACC Rankings ─────────────────────────────────────────────

def train_final_ranking_model(features: pl.DataFrame,
                              feature_cols: list[str],
                              target_col: str = "next_adj_eff_margin",
                              ) -> lgb.LGBMRegressor:
    """Train final ranking model on all available data."""
    X = features.select(feature_cols).to_pandas()
    y = features.select(target_col).to_pandas()[target_col]

    model = lgb.LGBMRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=10,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X, y)

    model.booster_.save_model(str(MODELS_DIR / "ranking_lgbm_final.txt"))
    print(f"Final ranking model saved to {MODELS_DIR / 'ranking_lgbm_final.txt'}")

    return model


def generate_acc_rankings(features: pl.DataFrame,
                          model: lgb.LGBMRegressor,
                          feature_cols: list[str],
                          prediction_season: int = 2025,
                          ) -> pd.DataFrame:
    """
    Generate predicted ACC rankings for the upcoming season.

    Uses the most recent season's features to predict next-season efficiency.
    Filters to current ACC membership.
    """
    # Get most recent season features for ACC teams
    latest = features.filter(
        (pl.col("year") == prediction_season) &
        (pl.col("team").is_in(CURRENT_ACC_TEAMS))
    )

    if len(latest) == 0:
        # Fall back to most recent available
        max_year = features.filter(
            pl.col("team").is_in(CURRENT_ACC_TEAMS)
        )["year"].max()
        latest = features.filter(
            (pl.col("year") == max_year) &
            (pl.col("team").is_in(CURRENT_ACC_TEAMS))
        )
        prediction_season = max_year
        print(f"  Using season {max_year} features (latest available for ACC)")

    X = latest.select(feature_cols).to_pandas()
    meta = latest.select(["team", "conf", "year", "adj_o", "adj_d",
                           "barthag", "returning_production_pct",
                           "coaching_tenure_years"]).to_pandas()

    # Predict
    predicted_eff_margin = model.predict(X)

    # Build rankings DataFrame
    rankings = meta.copy()
    rankings["predicted_eff_margin"] = np.round(predicted_eff_margin, 2)
    rankings["current_eff_margin"] = np.round(rankings["adj_o"] - rankings["adj_d"], 2)
    rankings["predicted_change"] = np.round(
        rankings["predicted_eff_margin"] - rankings["current_eff_margin"], 2
    )

    # Rank by predicted efficiency margin
    rankings = rankings.sort_values("predicted_eff_margin", ascending=False)
    rankings["predicted_rank"] = range(1, len(rankings) + 1)

    # Current rank for comparison
    rankings["current_rank"] = rankings["current_eff_margin"].rank(ascending=False).astype(int)
    rankings["rank_change"] = rankings["current_rank"] - rankings["predicted_rank"]

    return rankings


# ─── Visualization ───────────────────────────────────────────────────────────

def plot_actual_vs_predicted(cv_results: dict,
                             save_path: str = "plots/ranking_actual_vs_pred.png"):
    """Scatter plot of actual vs predicted efficiency margin."""
    fig, ax = plt.subplots(figsize=(8, 8))

    for fold in cv_results["folds"]:
        ax.scatter(fold["y_val"], fold["y_pred"],
                   alpha=0.3, s=20,
                   label=f"Fold {fold['fold']} (Val {fold['val_season']})")

    # Perfect prediction line
    all_vals = np.concatenate([
        np.concatenate([f["y_val"].values for f in cv_results["folds"]]),
        np.concatenate([f["y_pred"] for f in cv_results["folds"]]),
    ])
    lo, hi = all_vals.min() - 2, all_vals.max() + 2
    ax.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=1)

    ax.set_xlabel("Actual Efficiency Margin", fontsize=12)
    ax.set_ylabel("Predicted Efficiency Margin", fontsize=12)
    ax.set_title("Team Ranking Model — Actual vs Predicted", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Actual vs predicted plot saved to {save_path}")


def plot_acc_rankings(rankings: pd.DataFrame,
                      save_path: str = "plots/acc_rankings.png"):
    """Horizontal bar chart of predicted ACC rankings."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Sort by predicted rank (ascending for display bottom-to-top)
    df = rankings.sort_values("predicted_rank", ascending=True)

    colors = ["#22c55e" if x > 0 else "#ef4444" if x < 0 else "#6b7280"
              for x in df["predicted_eff_margin"]]

    bars = ax.barh(range(len(df)), df["predicted_eff_margin"], color=colors, height=0.7)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(
        [f"{row['predicted_rank']}. {row['team']}" for _, row in df.iterrows()],
        fontsize=11
    )
    ax.set_xlabel("Predicted Efficiency Margin", fontsize=12)
    ax.set_title("ACC 2025-26 Predicted Team Rankings\n(Current Membership)", fontsize=14)
    ax.axvline(x=0, color="gray", linewidth=0.8, linestyle="-")
    ax.grid(True, alpha=0.2, axis="x")

    # Add value labels
    for bar, val in zip(bars, df["predicted_eff_margin"]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f"{val:+.1f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"ACC rankings chart saved to {save_path}")


def plot_ranking_residuals(cv_results: dict,
                           save_path: str = "plots/ranking_residuals.png"):
    """Residual distribution plot."""
    all_y = np.concatenate([f["y_val"].values for f in cv_results["folds"]])
    all_pred = np.concatenate([f["y_pred"] for f in cv_results["folds"]])
    residuals = all_y - all_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residual distribution
    axes[0].hist(residuals, bins=40, color="#3b82f6", alpha=0.7, edgecolor="white")
    axes[0].axvline(x=0, color="red", linewidth=1.5, linestyle="--")
    axes[0].set_xlabel("Residual (Actual - Predicted)", fontsize=12)
    axes[0].set_ylabel("Count", fontsize=12)
    axes[0].set_title("Residual Distribution", fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # Residuals vs predicted
    axes[1].scatter(all_pred, residuals, alpha=0.2, s=15, color="#3b82f6")
    axes[1].axhline(y=0, color="red", linewidth=1.5, linestyle="--")
    axes[1].set_xlabel("Predicted Efficiency Margin", fontsize=12)
    axes[1].set_ylabel("Residual", fontsize=12)
    axes[1].set_title("Residuals vs Predicted", fontsize=14)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Residual plots saved to {save_path}")


def plot_shap_ranking(model: lgb.LGBMRegressor,
                      X: pd.DataFrame,
                      save_path: str = "plots/ranking_shap.png"):
    """SHAP summary for ranking model."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X.head(500))

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X.head(500), show=False, max_display=17)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Ranking SHAP plot saved to {save_path}")


# ─── Main Pipeline ───────────────────────────────────────────────────────────

def run_ranking_pipeline() -> dict:
    """
    Execute the full ranking model pipeline.
    """
    print("=" * 60)
    print("PART 2: CONFERENCE TEAM RANKING ENGINE")
    print("=" * 60)

    # 1. Features
    print("\n[1/6] Building features...")
    features = build_ranking_features()
    feature_cols = get_ranking_feature_columns()
    print(f"  {len(features)} team-seasons, {len(feature_cols)} features")

    # 2. Temporal CV
    print("\n[2/6] Running temporal cross-validation...")
    cv_results = temporal_cv_ranking(features, feature_cols)

    # 3. Plots
    print("\n[3/6] Generating evaluation plots...")
    plot_actual_vs_predicted(cv_results)
    plot_ranking_residuals(cv_results)

    # SHAP on last fold
    last_fold = cv_results["folds"][-1]
    plot_shap_ranking(last_fold["model"], last_fold["X_val"])

    # 4. Final model
    print("\n[4/6] Training final model on all data...")
    final_model = train_final_ranking_model(features, feature_cols)

    # 5. ACC Rankings
    print("\n[5/6] Generating ACC rankings...")
    rankings = generate_acc_rankings(features, final_model, feature_cols)
    print("\n  ACC Predicted Rankings (2025-26):")
    print("  " + "-" * 65)
    for _, row in rankings.iterrows():
        change = f"(+{int(row['rank_change'])})" if row['rank_change'] > 0 \
                 else f"(-{abs(int(row['rank_change']))})" if row['rank_change'] < 0 \
                 else "(—)"
        print(f"  {int(row['predicted_rank']):>2}. {row['team']:<20s} "
              f"Pred EM: {row['predicted_eff_margin']:>+6.1f}  "
              f"Curr EM: {row['current_eff_margin']:>+6.1f}  "
              f"{change}")

    plot_acc_rankings(rankings)
    rankings.to_csv(MODELS_DIR / "acc_rankings.csv", index=False)
    print(f"\n  Rankings saved to {MODELS_DIR / 'acc_rankings.csv'}")

    # 6. Save metrics
    print("\n[6/6] Saving metrics...")
    metrics = {
        "avg_mae": cv_results["avg_mae"],
        "avg_rmse": cv_results["avg_rmse"],
        "avg_r2": cv_results["avg_r2"],
        "n_team_seasons": len(features),
        "n_features": len(feature_cols),
        "folds": [
            {
                "fold": f["fold"],
                "val_season": f["val_season"],
                "n_val": f["n_val"],
                "mae": f["mae"],
                "rmse": f["rmse"],
                "r2": f["r2"],
            }
            for f in cv_results["folds"]
        ],
    }
    with open(MODELS_DIR / "ranking_model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return {
        "cv_results": cv_results,
        "final_model": final_model,
        "rankings": rankings,
        "feature_cols": feature_cols,
        "features": features,
    }


if __name__ == "__main__":
    run_ranking_pipeline()
