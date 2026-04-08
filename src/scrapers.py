"""
Real data scrapers for BartTorvik NCAA basketball data.

Scrapes team stats (JSON endpoint), player stats (CSV endpoint),
and derives transfer portal data by tracking player movement across seasons.

Usage:
    python src/scrapers.py --all         # Scrape everything
    python src/scrapers.py --players     # Player stats only
    python src/scrapers.py --teams       # Team stats only
    python src/scrapers.py --transfers   # Derive transfers from player data
"""

import argparse
import csv
import io
import time
import json
from pathlib import Path

import httpx
import numpy as np
import polars as pl


# ─── Configuration ───────────────────────────────────────────────────────────

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

SEASONS = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

REQUEST_DELAY_SEC = 2.0


def _get(url: str, client: httpx.Client) -> httpx.Response:
    """GET with rate limiting and retry."""
    time.sleep(REQUEST_DELAY_SEC)
    for attempt in range(3):
        try:
            resp = client.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return resp
        except (httpx.HTTPStatusError, httpx.ConnectError) as e:
            if attempt == 2:
                raise
            print(f"  Retry {attempt + 1} for {url}: {e}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Failed after 3 attempts: {url}")


def _height_to_inches(h: str) -> int:
    """Convert height string like '6-4' to inches (76)."""
    try:
        parts = h.strip().split("-")
        return int(parts[0]) * 12 + int(parts[1])
    except (ValueError, IndexError):
        return 77  # default ~6'5"


# ─── BartTorvik Team Stats (JSON endpoint) ─────────────────────────────────

def scrape_barttorvik_teams(seasons: list[int] = SEASONS) -> pl.DataFrame:
    """
    Scrape team-level stats from BartTorvik's JSON endpoint.

    Endpoint: https://barttorvik.com/{year}_team_results.json
    Returns a list of 45-element arrays per team (no keys).

    Column mapping (verified against known team stats):
      [1] team, [2] conf, [4] adj_o, [6] adj_d, [8] barthag,
      [10] rec_w, [11] rec_l, [15] sos, [32] wab, [44] adj_t
    """
    all_rows = []

    with httpx.Client() as client:
        for season in seasons:
            print(f"  Scraping team stats for {season}...")
            url = f"https://barttorvik.com/{season}_team_results.json"
            resp = _get(url, client)

            try:
                data = json.loads(resp.text)
            except json.JSONDecodeError:
                print(f"    WARNING: Invalid JSON for {season}")
                continue

            for entry in data:
                if not isinstance(entry, list) or len(entry) < 45:
                    continue

                try:
                    all_rows.append({
                        "team": str(entry[1]),
                        "conf": str(entry[2]),
                        "adj_o": round(float(entry[4]), 1),
                        "adj_d": round(float(entry[6]), 1),
                        "barthag": round(float(entry[8]), 4),
                        "adj_t": round(float(entry[44]), 1),
                        "rec_w": int(float(entry[10])),
                        "rec_l": int(float(entry[11])),
                        "wab": round(float(entry[32]), 1),
                        "sos": round(float(entry[15]), 4),
                        "sos_adj": round(float(entry[22]), 4),
                        "year": season,
                    })
                except (ValueError, TypeError, IndexError) as e:
                    continue  # skip malformed entries

            print(f"    Got {len([r for r in all_rows if r['year'] == season])} teams for {season}")

    if not all_rows:
        raise RuntimeError("No team data scraped. Check BartTorvik availability.")

    return pl.DataFrame(all_rows)


# ─── BartTorvik Player Stats (CSV endpoint) ────────────────────────────────

# Column indices for BartTorvik getadvstats.php (headerless CSV, 67 columns)
# Verified against Cooper Flagg (Duke), Boogie Fland (Arkansas), etc.
_P = {
    "player": 0,
    "team": 1,
    "conf": 2,
    "g": 3,
    "min_pct": 4,       # % of game minutes played (convert to mpg: * 40 / 100)
    "ortg": 5,
    "usg": 6,
    "efg": 7,            # percentage form (52.4 = 52.4%)
    "ts_pct": 8,          # percentage form
    "orb_pct": 9,
    "drb_pct": 10,
    "ast_pct": 11,
    "to_pct": 12,
    "blk_pct": 22,
    "stl_pct": 23,
    "ftr": 24,            # percentage form (42.9 = 42.9%)
    "class_year": 25,
    "height": 26,         # string "6-4"
    "porpag": 28,
    "year": 31,
    "pid": 32,
}


def scrape_barttorvik_players(seasons: list[int] = SEASONS) -> pl.DataFrame:
    """
    Scrape player-level stats from BartTorvik.

    Endpoint: https://barttorvik.com/getadvstats.php?year={year}&csv=1
    Returns headerless CSV with 67 columns per player.
    Uses csv.reader for proper handling of quoted fields.
    """
    all_rows = []

    with httpx.Client() as client:
        for season in seasons:
            print(f"  Scraping player stats for {season}...")
            url = f"https://barttorvik.com/getadvstats.php?year={season}&csv=1"
            resp = _get(url, client)

            # Skip if we got HTML (anti-bot page) instead of CSV
            if resp.text.strip().startswith("<"):
                print(f"    WARNING: Got HTML instead of CSV for {season}")
                continue

            reader = csv.reader(io.StringIO(resp.text))
            season_count = 0

            for row in reader:
                if len(row) < 33:
                    continue

                try:
                    min_pct = float(row[_P["min_pct"]])
                    mpg = round(min_pct * 40 / 100, 1)
                    ortg = float(row[_P["ortg"]])
                    usg = float(row[_P["usg"]])
                    g = int(float(row[_P["g"]]))

                    # Convert percentage-form stats to decimal where the
                    # synthetic data generator uses decimals
                    efg = float(row[_P["efg"]]) / 100
                    ts_pct = float(row[_P["ts_pct"]]) / 100
                    ftr = float(row[_P["ftr"]]) / 100

                    # These stay as percentages (matching synthetic data format)
                    orb_pct = float(row[_P["orb_pct"]])
                    drb_pct = float(row[_P["drb_pct"]])
                    ast_pct = float(row[_P["ast_pct"]])
                    to_pct = float(row[_P["to_pct"]])
                    blk_pct = float(row[_P["blk_pct"]])
                    stl_pct = float(row[_P["stl_pct"]])
                    porpag = float(row[_P["porpag"]])

                    # Compute BPM approximations (consistent formula for all players)
                    obpm = round((ortg - 105) / 10 * (usg / 20), 1)
                    dbpm = round((drb_pct - 14) / 5 + (blk_pct + stl_pct - 3) / 3, 1)
                    bpm = round(obpm + dbpm, 1)

                    height_in = _height_to_inches(row[_P["height"]])
                    class_year_raw = row[_P["class_year"]].strip()

                    all_rows.append({
                        "player": row[_P["player"]].strip(),
                        "team": row[_P["team"]].strip(),
                        "conf": row[_P["conf"]].strip(),
                        "g": g,
                        "mpg": mpg,
                        "ortg": round(ortg, 1),
                        "usg": round(usg, 1),
                        "efg": round(efg, 3),
                        "ts_pct": round(ts_pct, 3),
                        "ast_pct": round(ast_pct, 1),
                        "to_pct": round(to_pct, 1),
                        "orb_pct": round(orb_pct, 1),
                        "drb_pct": round(drb_pct, 1),
                        "blk_pct": round(blk_pct, 1),
                        "stl_pct": round(stl_pct, 1),
                        "ftr": round(ftr, 3),
                        "porpag": round(porpag, 2),
                        "bpm": bpm,
                        "obpm": obpm,
                        "dbpm": dbpm,
                        "year": season,
                        "pid": row[_P["pid"]].strip(),
                        "class_year": class_year_raw,
                        "height_in": height_in,
                        "recruiting_stars": 2,  # default; 247 data unavailable
                    })
                    season_count += 1

                except (ValueError, IndexError):
                    continue

            print(f"    Got {season_count} players for {season}")

    if not all_rows:
        raise RuntimeError("No player data scraped. Check BartTorvik availability.")

    return pl.DataFrame(all_rows)


# ─── Transfer Derivation (from player data across seasons) ──────────────────

def derive_transfers(
    player_stats_path: Path = DATA_DIR / "player_stats.parquet",
    team_stats_path: Path = DATA_DIR / "team_stats.parquet",
) -> pl.DataFrame:
    """
    Derive transfer portal data by tracking player movement across seasons.

    A transfer is detected when a player (by pid) appears on a different
    team in consecutive seasons. Pre-transfer stats come from the origin
    season, post-transfer stats from the destination season.

    This replaces VerbalCommits scraping — more reliable and automatically
    provides matched pre/post stats with no fuzzy name matching needed.
    """
    players = pl.read_parquet(player_stats_path)
    teams = pl.read_parquet(team_stats_path)

    # Build team stats lookup: (team, year) -> stats
    team_lookup = {}
    for row in teams.iter_rows(named=True):
        team_lookup[(row["team"], row["year"])] = row

    # Group players by pid to find team changes
    seasons = sorted(players["year"].unique().to_list())
    pid_seasons = {}  # pid -> list of (year, row_dict)

    for row in players.iter_rows(named=True):
        pid = row["pid"]
        if pid not in pid_seasons:
            pid_seasons[pid] = []
        pid_seasons[pid].append((row["year"], row))

    transfer_rows = []
    stat_fields = [
        "g", "mpg", "ortg", "usg", "efg", "ts_pct", "ast_pct", "to_pct",
        "orb_pct", "drb_pct", "blk_pct", "stl_pct", "ftr", "porpag",
        "bpm", "obpm", "dbpm",
    ]

    for pid, entries in pid_seasons.items():
        entries.sort(key=lambda x: x[0])

        for i in range(len(entries) - 1):
            pre_year, pre = entries[i]
            post_year, post = entries[i + 1]

            # Must be consecutive seasons and different teams
            if post_year != pre_year + 1:
                continue
            if pre["team"] == post["team"]:
                continue

            # Get team context
            origin_ts = team_lookup.get((pre["team"], pre_year))
            dest_ts = team_lookup.get((post["team"], post_year))
            if origin_ts is None or dest_ts is None:
                continue

            # Build transfer record matching synthetic data schema
            rec = {
                "player": pre["player"],
                "pid": pid,
                "origin_team": pre["team"],
                "origin_conf": pre["conf"],
                "dest_team": post["team"],
                "dest_conf": post["conf"],
                "transfer_season": post_year,
                "recruiting_stars": pre.get("recruiting_stars", 2),
                "class_year": pre.get("class_year", "So"),
                "height_in": pre.get("height_in", 77),
            }

            # Pre-transfer stats
            for f in stat_fields:
                rec[f"pre_{f}"] = pre[f]

            # Post-transfer stats
            for f in stat_fields:
                rec[f"post_{f}"] = post[f]

            # Team context
            rec["origin_adj_o"] = origin_ts["adj_o"]
            rec["origin_adj_d"] = origin_ts["adj_d"]
            rec["origin_adj_t"] = origin_ts["adj_t"]
            rec["origin_barthag"] = origin_ts["barthag"]
            rec["dest_adj_o"] = dest_ts["adj_o"]
            rec["dest_adj_d"] = dest_ts["adj_d"]
            rec["dest_adj_t"] = dest_ts["adj_t"]
            rec["dest_barthag"] = dest_ts["barthag"]

            # Target: success = post ortg >= pre ortg
            rec["success"] = int(post["ortg"] >= pre["ortg"])

            transfer_rows.append(rec)

    print(f"  Derived {len(transfer_rows)} transfers from player data")
    return pl.DataFrame(transfer_rows) if transfer_rows else pl.DataFrame()


# ─── Derived Data Builders ──────────────────────────────────────────────────

def build_coaching_data(team_stats_path: Path = DATA_DIR / "team_stats.parquet") -> pl.DataFrame:
    """
    Derive coaching tenure from BartTorvik team stats.

    If the team stats included a coach column, tenure is computed by
    tracking consecutive seasons with the same coach per team.
    Otherwise, a default tenure of 5 years (D1 median) is used.
    """
    teams = pl.read_parquet(team_stats_path)

    if "coach" in teams.columns:
        print("  Found coach column — computing tenure from coach names")
        rows = []
        for team_name in sorted(teams["team"].unique().to_list()):
            team_rows = (
                teams.filter(pl.col("team") == team_name)
                .sort("year")
            )
            tenure = 0
            prev_coach = None
            for row in team_rows.iter_rows(named=True):
                coach = str(row["coach"]).strip().lower()
                if coach == prev_coach:
                    tenure += 1
                else:
                    tenure = 1
                    prev_coach = coach
                rows.append({
                    "team": team_name,
                    "year": row["year"],
                    "coaching_tenure_years": tenure,
                })
        return pl.DataFrame(rows)

    print("  NOTE: No coach column in team stats — using default tenure (5 years)")
    return teams.select(["team", "year"]).unique().with_columns(
        pl.lit(5).alias("coaching_tenure_years")
    )


def build_returning_production(
    player_stats_path: Path = DATA_DIR / "player_stats.parquet",
    transfers_path: Path = DATA_DIR / "transfers.parquet",
) -> pl.DataFrame:
    """
    Compute returning production and transfer activity per team-season.

    Returning production = fraction of prior season's total minutes
    played by players who return to the same team next season.
    Incoming transfers are counted from the derived transfer data.
    """
    players = pl.read_parquet(player_stats_path)

    players = players.with_columns([
        (pl.col("mpg").cast(pl.Float64) * pl.col("g").cast(pl.Float64)).alias("total_min"),
    ])

    # Team total minutes per season
    team_minutes = players.group_by(["team", "year"]).agg(
        pl.col("total_min").sum().alias("team_total_min")
    )

    # Find returning players by pid: same pid + same team in consecutive seasons
    prior = players.select(["pid", "team", "year", "total_min"])
    next_yr = players.select([
        "pid", "team",
        (pl.col("year") - 1).alias("prior_year"),
    ]).unique()

    returning = prior.join(
        next_yr,
        left_on=["pid", "team", "year"],
        right_on=["pid", "team", "prior_year"],
        how="inner",
    )

    # Aggregate returning minutes, mapped to the NEXT season
    ret_agg = returning.with_columns(
        (pl.col("year") + 1).alias("next_season")
    ).group_by(["team", "next_season"]).agg(
        pl.col("total_min").sum().alias("returning_min")
    )

    # Join with prior season's team totals
    team_min_shifted = team_minutes.with_columns(
        (pl.col("year") + 1).alias("next_season")
    ).select(["team", "next_season", "team_total_min"])

    result = ret_agg.join(
        team_min_shifted,
        on=["team", "next_season"],
        how="inner",
    ).with_columns(
        (pl.col("returning_min") / pl.col("team_total_min").clip(lower_bound=1.0))
        .clip(0.0, 1.0)
        .round(3)
        .alias("returning_production_pct")
    ).rename({"next_season": "year"}).select(["team", "year", "returning_production_pct"])

    # Add incoming transfer counts
    if transfers_path.exists():
        transfers = pl.read_parquet(transfers_path)
        transfer_counts = transfers.group_by(["dest_team", "transfer_season"]).agg(
            pl.len().alias("n_transfers_in")
        ).rename({"dest_team": "team", "transfer_season": "year"})

        result = result.join(transfer_counts, on=["team", "year"], how="left")
        result = result.with_columns(pl.col("n_transfers_in").fill_null(0))
    else:
        print("  NOTE: No transfers.parquet found — setting n_transfers_in to 0")
        result = result.with_columns(pl.lit(0).alias("n_transfers_in"))

    # Transfer composite: count × avg quality estimate
    result = result.with_columns(
        (pl.col("n_transfers_in").cast(pl.Float64) * 2.5).alias("incoming_transfer_composite")
    )

    return result


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NCAA Basketball Data Scrapers")
    parser.add_argument("--all", action="store_true", help="Scrape all sources")
    parser.add_argument("--players", action="store_true", help="Player stats only")
    parser.add_argument("--teams", action="store_true", help="Team stats only")
    parser.add_argument("--transfers", action="store_true", help="Derive transfers from player data")
    parser.add_argument("--coaching", action="store_true", help="Build coaching tenure data")
    parser.add_argument("--returning", action="store_true", help="Build returning production data")
    args = parser.parse_args()

    if args.all or args.teams:
        print("\n=== Scraping BartTorvik Team Stats ===")
        teams = scrape_barttorvik_teams()
        teams.write_parquet(DATA_DIR / "team_stats.parquet")
        print(f"Saved {len(teams)} team-seasons")

    if args.all or args.players:
        print("\n=== Scraping BartTorvik Player Stats ===")
        players = scrape_barttorvik_players()
        players.write_parquet(DATA_DIR / "player_stats.parquet")
        print(f"Saved {len(players)} player-seasons")

    if args.all or args.transfers:
        print("\n=== Deriving Transfers from Player Data ===")
        transfers = derive_transfers()
        transfers.write_parquet(DATA_DIR / "transfers.parquet")
        print(f"Saved {len(transfers)} matched transfers")

    if args.all or args.coaching:
        print("\n=== Building Coaching Tenure Data ===")
        coaching = build_coaching_data()
        coaching.write_parquet(DATA_DIR / "coaching.parquet")
        print(f"Saved {len(coaching)} coaching records")

    if args.all or args.returning:
        print("\n=== Building Returning Production Data ===")
        ret_prod = build_returning_production()
        ret_prod.write_parquet(DATA_DIR / "returning_production.parquet")
        print(f"Saved {len(ret_prod)} returning production records")


if __name__ == "__main__":
    main()
