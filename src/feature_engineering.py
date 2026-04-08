"""
Feature engineering for both the Transfer Success Model and Team Ranking Engine.

Ingestions and transformations uses Polars.
Output DataFrames are converted to pandas only at the model boundary
for scikit-learn/XGBoost compatibility.
"""

import polars as pl
import numpy as np
from pathlib import Path


DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)


# ─── Transfer Success Features ───────────────────────────────────────────────

def build_transfer_features(transfers_path: str = "data/raw/transfers.parquet",
                            team_stats_path: str = "data/raw/team_stats.parquet",
                            ) -> pl.DataFrame:
    """
    Build feature matrix for Transfer Success Prediction Model.

    Feature categories:
    1. Pre-transfer individual performance (ortg, usg, bpm, efg, etc.)
    2. Origin team context (adj_o, adj_d, barthag, conference)
    3. Destination team context (adj_o, adj_d, barthag, conference)
    4. Contextual deltas (destination - origin quality, conference jump)
    5. Player profile (recruiting stars, class year, height)
    6. Interaction features (usage × team quality, mpg × conference jump)

    Target: success (1 = post-transfer ortg >= pre-transfer ortg)
    """
    transfers = pl.read_parquet(transfers_path)
    team_stats = pl.read_parquet(team_stats_path)

    # Handle columns that may be missing with real scraped data
    # (247Sports recruiting data cannot be reliably scraped)
    if "recruiting_stars" not in transfers.columns:
        transfers = transfers.with_columns(pl.lit(2).alias("recruiting_stars"))
    else:
        transfers = transfers.with_columns(
            pl.col("recruiting_stars").fill_null(2)
        )
    if "height_in" not in transfers.columns:
        transfers = transfers.with_columns(pl.lit(77).alias("height_in"))
    else:
        transfers = transfers.with_columns(
            pl.col("height_in").fill_null(77)
        )
    if "class_year" not in transfers.columns:
        transfers = transfers.with_columns(pl.lit("So").alias("class_year"))
    else:
        transfers = transfers.with_columns(
            pl.col("class_year").fill_null("So")
        )

    # ── Derived features ──

    features = transfers.with_columns([
        # Team quality deltas
        (pl.col("dest_adj_o") - pl.col("origin_adj_o")).alias("delta_adj_o"),
        (pl.col("dest_adj_d") - pl.col("origin_adj_d")).alias("delta_adj_d"),
        (pl.col("dest_barthag") - pl.col("origin_barthag")).alias("delta_barthag"),
        (pl.col("dest_adj_t") - pl.col("origin_adj_t")).alias("delta_tempo"),

        # Conference jump indicator (power conf = ACC, B10, SEC, B12, BE)
        pl.col("origin_conf").is_in(["ACC", "B10", "SEC", "B12", "BE"])
        .cast(pl.Int8).alias("origin_is_power"),
        pl.col("dest_conf").is_in(["ACC", "B10", "SEC", "B12", "BE"])
        .cast(pl.Int8).alias("dest_is_power"),

        # Pre-transfer efficiency composite
        (pl.col("pre_ortg") * pl.col("pre_usg") / 100).alias("pre_ortg_usg_product"),

        # Minutes share (proxy for role)
        (pl.col("pre_mpg") / 40).alias("pre_minutes_share"),

        # Assist-to-turnover ratio
        (pl.col("pre_ast_pct") / pl.col("pre_to_pct").clip(lower_bound=1))
        .alias("pre_ast_to_ratio"),

        # Shooting efficiency composite (eFG weighted with FTR)
        (pl.col("pre_efg") + 0.1 * pl.col("pre_ftr")).alias("pre_shooting_composite"),

        # Rebound rate combined
        (pl.col("pre_orb_pct") + pl.col("pre_drb_pct")).alias("pre_total_reb_pct"),

        # Defensive contribution proxy
        (pl.col("pre_blk_pct") + pl.col("pre_stl_pct")).alias("pre_defensive_stocks"),
    ])

    # Conference jump direction
    features = features.with_columns([
        (pl.col("dest_is_power") - pl.col("origin_is_power")).alias("conf_jump_direction"),
    ])

    # Interaction features
    features = features.with_columns([
        (pl.col("pre_usg") * pl.col("delta_barthag")).alias("usg_x_quality_delta"),
        (pl.col("pre_mpg") * pl.col("conf_jump_direction").cast(pl.Float64))
        .alias("mpg_x_conf_jump"),
        (pl.col("pre_bpm") * pl.col("delta_adj_o")).alias("bpm_x_off_delta"),
    ])

    # Encode class year ordinally
    class_map = {"Fr": 1, "So": 2, "Jr": 3, "Sr": 4}
    features = features.with_columns(
        pl.col("class_year").replace_strict(class_map, default=2)
        .cast(pl.Int8).alias("class_year_ord")
    )

    return features


def get_transfer_feature_columns() -> list[str]:
    """Return the list of feature columns for the transfer model."""
    return [
        # Pre-transfer individual stats
        "pre_ortg", "pre_usg", "pre_efg", "pre_ts_pct",
        "pre_ast_pct", "pre_to_pct", "pre_orb_pct", "pre_drb_pct",
        "pre_blk_pct", "pre_stl_pct", "pre_ftr", "pre_porpag",
        "pre_bpm", "pre_obpm", "pre_dbpm", "pre_mpg", "pre_g",
        # Player profile
        "recruiting_stars", "class_year_ord", "height_in",
        # Origin team context
        "origin_adj_o", "origin_adj_d", "origin_adj_t", "origin_barthag",
        "origin_is_power",
        # Destination team context
        "dest_adj_o", "dest_adj_d", "dest_adj_t", "dest_barthag",
        "dest_is_power",
        # Deltas
        "delta_adj_o", "delta_adj_d", "delta_barthag", "delta_tempo",
        "conf_jump_direction",
        # Engineered features
        "pre_ortg_usg_product", "pre_minutes_share", "pre_ast_to_ratio",
        "pre_shooting_composite", "pre_total_reb_pct", "pre_defensive_stocks",
        # Interactions
        "usg_x_quality_delta", "mpg_x_conf_jump", "bpm_x_off_delta",
    ]


# ─── Team Ranking Features ──────────────────────────────────────────────────

def build_ranking_features(
    team_stats_path: str = "data/raw/team_stats.parquet",
    coaching_path: str = "data/raw/coaching.parquet",
    ret_prod_path: str = "data/raw/returning_production.parquet",
) -> pl.DataFrame:
    """
    Build feature matrix for Conference Team Ranking Engine.

    For each team-season, predict NEXT season's adjusted efficiency margin.
    Features from season N predict adj_eff_margin in season N+1.

    Feature categories:
    1. Prior season efficiency metrics (adj_o, adj_d, barthag, adj_t)
    2. Returning production (% of minutes returning, incoming transfers)
    3. Coaching stability (tenure years)
    4. Schedule strength (sos, sos_adj)
    5. Momentum features (win trajectory, conference performance)

    Target: next_adj_eff_margin (adj_o - adj_d for season N+1)
    """
    teams = pl.read_parquet(team_stats_path)
    coaching = pl.read_parquet(coaching_path)
    ret_prod = pl.read_parquet(ret_prod_path)

    # Compute efficiency margin
    teams = teams.with_columns(
        (pl.col("adj_o") - pl.col("adj_d")).alias("adj_eff_margin"),
        (pl.col("rec_w") / (pl.col("rec_w") + pl.col("rec_l"))).alias("win_pct"),
    )

    # Create next-season target by self-joining
    next_season = teams.select([
        pl.col("team"),
        pl.col("year"),
        pl.col("adj_eff_margin").alias("next_adj_eff_margin"),
        pl.col("adj_o").alias("next_adj_o"),
        pl.col("adj_d").alias("next_adj_d"),
    ]).with_columns(
        (pl.col("year") - 1).alias("prior_year")
    )

    # Join: current season features → next season target
    features = teams.join(
        next_season.select([
            "team", "prior_year", "next_adj_eff_margin", "next_adj_o", "next_adj_d"
        ]),
        left_on=["team", "year"],
        right_on=["team", "prior_year"],
        how="inner",
    )

    # Join coaching data
    features = features.join(coaching, on=["team", "year"], how="left")

    # Join returning production
    # Returning production for the NEXT season is what we need
    ret_prod_next = ret_prod.with_columns(
        (pl.col("year") - 1).alias("prior_year")
    ).select([
        "team", "prior_year",
        "returning_production_pct", "n_transfers_in", "incoming_transfer_composite",
    ])

    features = features.join(
        ret_prod_next,
        left_on=["team", "year"],
        right_on=["team", "prior_year"],
        how="left",
    )

    # Fill nulls in returning production
    features = features.with_columns([
        pl.col("returning_production_pct").fill_null(0.55),
        pl.col("n_transfers_in").fill_null(2),
        pl.col("incoming_transfer_composite").fill_null(5.0),
        pl.col("coaching_tenure_years").fill_null(3),
    ])

    # ── Engineered features ──

    features = features.with_columns([
        # Efficiency margin (current season)
        (pl.col("adj_o") - pl.col("adj_d")).alias("eff_margin"),

        # Is power conference
        pl.col("conf").is_in(["ACC", "B10", "SEC", "B12", "BE"])
        .cast(pl.Int8).alias("is_power_conf"),

        # Coaching stability bucket
        pl.when(pl.col("coaching_tenure_years") <= 2).then(pl.lit("new"))
        .when(pl.col("coaching_tenure_years") <= 5).then(pl.lit("establishing"))
        .otherwise(pl.lit("established"))
        .alias("coaching_stability"),

        # Portal activity intensity
        (pl.col("incoming_transfer_composite") / pl.col("n_transfers_in").clip(lower_bound=1))
        .alias("avg_transfer_quality"),

        # Returning production × prior efficiency interaction
        (pl.col("returning_production_pct") * pl.col("barthag"))
        .alias("returning_x_quality"),
    ])

    # Encode coaching stability
    stability_map = {"new": 1, "establishing": 2, "established": 3}
    features = features.with_columns(
        pl.col("coaching_stability").replace_strict(stability_map, default=2)
        .cast(pl.Int8).alias("coaching_stability_ord")
    )

    return features


def get_ranking_feature_columns() -> list[str]:
    """Return the list of feature columns for the ranking model."""
    return [
        # Prior season efficiency
        "adj_o", "adj_d", "adj_t", "barthag", "eff_margin",
        "wab", "win_pct", "sos", "sos_adj",
        # Returning production
        "returning_production_pct", "n_transfers_in",
        "incoming_transfer_composite", "avg_transfer_quality",
        # Coaching
        "coaching_tenure_years", "coaching_stability_ord",
        # Conference context
        "is_power_conf",
        # Interactions
        "returning_x_quality",
    ]


# ─── Temporal Split Utilities ────────────────────────────────────────────────

def temporal_split_transfers(features: pl.DataFrame,
                             train_through_season: int,
                             feature_cols: list[str],
                             target_col: str = "success",
                             ) -> tuple:
    """
    Temporal train/validation split for transfer model.

    Train on transfers through `train_through_season`,
    validate on the next season.

    Returns: (X_train, y_train, X_val, y_val) as pandas DataFrames.
    """
    train = features.filter(pl.col("transfer_season") <= train_through_season)
    val = features.filter(pl.col("transfer_season") == train_through_season + 1)

    X_train = train.select(feature_cols).to_pandas()
    y_train = train.select(target_col).to_pandas()[target_col]
    X_val = val.select(feature_cols).to_pandas()
    y_val = val.select(target_col).to_pandas()[target_col]

    return X_train, y_train, X_val, y_val


def temporal_split_ranking(features: pl.DataFrame,
                           train_through_season: int,
                           feature_cols: list[str],
                           target_col: str = "next_adj_eff_margin",
                           ) -> tuple:
    """
    Temporal train/validation split for ranking model.

    Uses season N features to predict season N+1 efficiency.
    """
    train = features.filter(pl.col("year") <= train_through_season)
    val = features.filter(pl.col("year") == train_through_season + 1)

    X_train = train.select(feature_cols).to_pandas()
    y_train = train.select(target_col).to_pandas()[target_col]
    X_val = val.select(feature_cols).to_pandas()
    y_val = val.select(target_col).to_pandas()[target_col]

    meta_val = val.select(["team", "conf", "year"]).to_pandas()

    return X_train, y_train, X_val, y_val, meta_val


# ─── Data Pipeline Orchestrator ──────────────────────────────────────────────

def prepare_all_features() -> dict:
    """
    Run the full feature engineering pipeline.
    Saves processed datasets and returns them.
    """
    print("Building transfer features...")
    transfer_features = build_transfer_features()
    transfer_features.write_parquet(DATA_PROCESSED / "transfer_features.parquet")
    print(f"  Transfer features: {transfer_features.shape}")
    print(f"  Success rate: {transfer_features['success'].mean():.3f}")

    print("\nBuilding ranking features...")
    ranking_features = build_ranking_features()
    ranking_features.write_parquet(DATA_PROCESSED / "ranking_features.parquet")
    print(f"  Ranking features: {ranking_features.shape}")

    return {
        "transfer_features": transfer_features,
        "ranking_features": ranking_features,
    }


if __name__ == "__main__":
    prepare_all_features()
