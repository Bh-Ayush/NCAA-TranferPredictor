"""
Part 1 — Transfer Success Prediction Model

XGBoost classifier predicting whether a transfer improves or maintains
their per-40-minute adjusted offensive rating at the destination school.

Evaluation:
- PR-AUC (primary metric — handles class imbalance)
- Calibration curve
- Temporal cross-validation (train through season N, validate on N+1)
- SHAP feature importance
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import shap

from src.feature_engineering import (
    build_transfer_features,
    get_transfer_feature_columns,
    temporal_split_transfers,
)


MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)


# ─── Temporal Cross-Validation ───────────────────────────────────────────────

def temporal_cv(features: pl.DataFrame,
                feature_cols: list[str],
                target_col: str = "success",
                ) -> dict:
    """
    Walk-forward temporal cross-validation.

    Requires at least 2 training seasons per fold. With seasons 2019-2025:
      Fold 1: Train 2019-2020, Validate 2021
      Fold 2: Train 2019-2021, Validate 2022
      ...etc, growing the training window each fold.

    Returns metrics per fold and aggregated.
    """
    seasons = sorted(features["transfer_season"].unique().to_list())
    # Start from index 1 so fold 1 trains on at least 2 seasons
    start_idx = min(1, len(seasons) - 2)
    n_folds = len(seasons) - 1 - start_idx
    print(f"Available transfer seasons: {seasons}")
    print(f"Temporal CV folds: {n_folds} (requiring >=2 training seasons)")

    fold_results = []

    for i in range(start_idx, len(seasons) - 1):
        train_through = seasons[i]
        val_season = seasons[i + 1]

        X_train, y_train, X_val, y_val = temporal_split_transfers(
            features, train_through, feature_cols, target_col
        )

        fold_num = i - start_idx + 1
        print(f"\n--- Fold {fold_num}: Train <={train_through}, Val={val_season} ---")
        print(f"  Train: {len(X_train)} samples, success rate: {y_train.mean():.3f}")
        print(f"  Val:   {len(X_val)} samples, success rate: {y_val.mean():.3f}")

        if len(X_val) < 10:
            print("  SKIP: Too few validation samples")
            continue

        # Train XGBoost
        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=y_train.value_counts().iloc[0] / max(y_train.value_counts().iloc[1], 1),
            eval_metric="aucpr",
            early_stopping_rounds=30,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Predictions
        y_prob = model.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        # Metrics
        pr_auc = average_precision_score(y_val, y_prob)
        roc_auc = roc_auc_score(y_val, y_prob)
        brier = brier_score_loss(y_val, y_prob)

        print(f"  PR-AUC:  {pr_auc:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"  Brier:   {brier:.4f}")

        fold_results.append({
            "fold": fold_num,
            "train_through": train_through,
            "val_season": val_season,
            "n_train": len(X_train),
            "n_val": len(X_val),
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "brier_score": brier,
            "model": model,
            "y_val": y_val,
            "y_prob": y_prob,
            "y_pred": y_pred,
            "X_val": X_val,
        })

    # Aggregate metrics
    avg_pr_auc = np.mean([f["pr_auc"] for f in fold_results])
    avg_roc_auc = np.mean([f["roc_auc"] for f in fold_results])
    avg_brier = np.mean([f["brier_score"] for f in fold_results])

    print(f"\n=== Aggregated Temporal CV Results ===")
    print(f"  Mean PR-AUC:  {avg_pr_auc:.4f}")
    print(f"  Mean ROC-AUC: {avg_roc_auc:.4f}")
    print(f"  Mean Brier:   {avg_brier:.4f}")

    return {
        "folds": fold_results,
        "avg_pr_auc": avg_pr_auc,
        "avg_roc_auc": avg_roc_auc,
        "avg_brier": avg_brier,
    }


# ─── Final Model Training ───────────────────────────────────────────────────

def train_final_model(features: pl.DataFrame,
                      feature_cols: list[str],
                      target_col: str = "success",
                      ) -> xgb.XGBClassifier:
    """
    Train the final model on ALL available data.
    Used for deployment / Streamlit app.
    """
    X = features.select(feature_cols).to_pandas()
    y = features.select(target_col).to_pandas()[target_col]

    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=y.value_counts().iloc[0] / max(y.value_counts().iloc[1], 1),
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y, verbose=False)

    # Save
    model.save_model(str(MODELS_DIR / "transfer_xgb_final.json"))
    print(f"Final model saved to {MODELS_DIR / 'transfer_xgb_final.json'}")

    return model


# ─── Visualization ───────────────────────────────────────────────────────────

def plot_pr_curve(cv_results: dict, save_path: str = "plots/transfer_pr_curve.png"):
    """Plot Precision-Recall curve for each CV fold."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    for fold in cv_results["folds"]:
        precision, recall, _ = precision_recall_curve(fold["y_val"], fold["y_prob"])
        ax.plot(recall, precision,
                label=f"Fold {fold['fold']} (Val {fold['val_season']}, "
                      f"PR-AUC={fold['pr_auc']:.3f})",
                linewidth=2)

    # Baseline: class prevalence
    all_y = pd.concat([f["y_val"] for f in cv_results["folds"]])
    baseline = all_y.mean()
    ax.axhline(y=baseline, color="gray", linestyle="--", label=f"Baseline ({baseline:.2f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Transfer Success Prediction — PR Curve (Temporal CV)", fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"PR curve saved to {save_path}")


def plot_calibration(cv_results: dict, save_path: str = "plots/transfer_calibration.png"):
    """Plot calibration curve (reliability diagram)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Combine all folds
    all_y = pd.concat([f["y_val"] for f in cv_results["folds"]])
    all_prob = np.concatenate([f["y_prob"] for f in cv_results["folds"]])

    # Calibration curve
    prob_true, prob_pred = calibration_curve(all_y, all_prob, n_bins=10, strategy="uniform")
    axes[0].plot(prob_pred, prob_true, "s-", color="#2563eb", linewidth=2, markersize=8)
    axes[0].plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    axes[0].set_xlabel("Mean Predicted Probability", fontsize=12)
    axes[0].set_ylabel("Fraction of Positives", fontsize=12)
    axes[0].set_title("Calibration Curve", fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # Prediction distribution
    axes[1].hist(all_prob[all_y == 1], bins=30, alpha=0.6, label="Success", color="#22c55e")
    axes[1].hist(all_prob[all_y == 0], bins=30, alpha=0.6, label="Decline", color="#ef4444")
    axes[1].set_xlabel("Predicted Probability", fontsize=12)
    axes[1].set_ylabel("Count", fontsize=12)
    axes[1].set_title("Prediction Distribution", fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Calibration plot saved to {save_path}")


def plot_shap_importance(model: xgb.XGBClassifier,
                         X: pd.DataFrame,
                         save_path: str = "plots/transfer_shap.png"):
    """SHAP summary plot for feature importance."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X.head(500))  # Sample for speed

    fig, ax = plt.subplots(figsize=(10, 10))
    shap.summary_plot(shap_values, X.head(500), show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP plot saved to {save_path}")


def plot_confusion_matrix(cv_results: dict,
                          save_path: str = "plots/transfer_confusion.png"):
    """Confusion matrix from combined CV folds."""
    all_y = pd.concat([f["y_val"] for f in cv_results["folds"]])
    all_pred = np.concatenate([f["y_pred"] for f in cv_results["folds"]])

    cm = confusion_matrix(all_y, all_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Decline", "Success"],
                yticklabels=["Decline", "Success"])
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Transfer Success — Confusion Matrix (All Folds)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


# ─── Full Classification Report ──────────────────────────────────────────────

def full_classification_report(cv_results: dict) -> str:
    """Generate full classification report from combined CV folds."""
    all_y = pd.concat([f["y_val"] for f in cv_results["folds"]])
    all_pred = np.concatenate([f["y_pred"] for f in cv_results["folds"]])

    report = classification_report(
        all_y, all_pred,
        target_names=["Decline", "Success"],
        digits=4,
    )
    return report


# ─── Main Pipeline ───────────────────────────────────────────────────────────

def run_transfer_model_pipeline() -> dict:
    """
    Execute the full transfer success prediction pipeline.

    Steps:
    1. Load and engineer features
    2. Run temporal cross-validation
    3. Generate evaluation plots
    4. Train final model
    5. Save results

    Returns dict with cv_results, final_model, and feature_cols.
    """
    print("=" * 60)
    print("PART 1: TRANSFER SUCCESS PREDICTION MODEL")
    print("=" * 60)

    # 1. Features
    print("\n[1/5] Building features...")
    features = build_transfer_features()
    feature_cols = get_transfer_feature_columns()
    print(f"  {len(features)} transfers, {len(feature_cols)} features")

    # 2. Temporal CV
    print("\n[2/5] Running temporal cross-validation...")
    cv_results = temporal_cv(features, feature_cols)

    # 3. Classification report
    print("\n[3/5] Classification report:")
    report = full_classification_report(cv_results)
    print(report)

    # 4. Plots
    print("\n[4/5] Generating evaluation plots...")
    plot_pr_curve(cv_results)
    plot_calibration(cv_results)
    plot_confusion_matrix(cv_results)

    # SHAP on the last fold's model
    last_fold = cv_results["folds"][-1]
    plot_shap_importance(last_fold["model"], last_fold["X_val"])

    # 5. Final model
    print("\n[5/5] Training final model on all data...")
    final_model = train_final_model(features, feature_cols)

    # Save metrics
    metrics = {
        "avg_pr_auc": cv_results["avg_pr_auc"],
        "avg_roc_auc": cv_results["avg_roc_auc"],
        "avg_brier": cv_results["avg_brier"],
        "n_transfers": len(features),
        "n_features": len(feature_cols),
        "folds": [
            {
                "fold": f["fold"],
                "val_season": f["val_season"],
                "n_val": f["n_val"],
                "pr_auc": f["pr_auc"],
                "roc_auc": f["roc_auc"],
                "brier_score": f["brier_score"],
            }
            for f in cv_results["folds"]
        ],
    }
    with open(MODELS_DIR / "transfer_model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {MODELS_DIR / 'transfer_model_metrics.json'}")

    return {
        "cv_results": cv_results,
        "final_model": final_model,
        "feature_cols": feature_cols,
        "features": features,
    }


if __name__ == "__main__":
    run_transfer_model_pipeline()
