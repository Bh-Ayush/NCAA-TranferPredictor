"""
Synthetic data generator for NCAA Basketball Transfer Portal project produced by Claude.

Generates realistic data mirroring BartTorvik player-level and team-level schemas.
Used for development and demonstration ONLY when live API access is unavailable.
Replace with real data via src/scrapers.py for production use.

BartTorvik columns reference:
  Player-level: player, team, conf, g, mpg, ortg, usg, efg, ts_pct, ast_pct,
                to_pct, orb_pct, drb_pct, blk_pct, stl_pct, ftr, porpag,
                bpm, obpm, dbpm, year, pid
  Team-level:   team, conf, barthag, adj_o, adj_d, adj_t, wab, seed,
                year, sos, sos_adj
"""

import numpy as np
import polars as pl
import hashlib
from typing import Optional


# ─── Constants ───────────────────────────────────────────────────────────────

CURRENT_ACC_TEAMS = [
    "Florida St.", "Clemson", "North Carolina", "Duke", "Virginia",
    "Louisville", "Pittsburgh", "NC State", "Wake Forest", "Syracuse",
    "Georgia Tech", "Boston College", "Notre Dame", "Miami FL",
    "Virginia Tech", "California", "Stanford", "SMU",
]

ALL_D1_CONFERENCES = [
    "ACC", "B10", "B12", "SEC", "BE", "P12", "A10", "MWC", "WCC",
    "AAC", "MVC", "CAA", "MAC", "SBC", "CUSA", "Amer", "SC", "OVC",
    "Horz", "Sum", "WAC", "BSky", "BW", "Ivy", "Pat", "MAAC",
    "NEC", "MEAC", "SWAC", "ASun", "Ind",
]

SEASONS = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

# Realistic name pools
FIRST_NAMES = [
    "Jaylen", "Marcus", "DeAndre", "Caleb", "Isaiah", "Tyrese", "Cameron",
    "Jalen", "Terrence", "Ahmad", "Brandon", "Malik", "Kevin", "Darius",
    "Robert", "Xavier", "Michael", "Tyler", "Jordan", "Chris", "Dalton",
    "Armando", "RJ", "AJ", "DJ", "TJ", "PJ", "JJ", "Tre", "Kam",
    "Zach", "Ryan", "Noah", "Hunter", "Cole", "Riley", "Seth", "Luke",
    "Ethan", "Connor", "Grant", "Kyle", "Drew", "Ben", "Jake", "Matt",
    "Will", "Sam", "Alex", "Nick", "Jack", "Trey", "Quincy", "Devin",
    "Jamal", "Andre", "Rasheed", "Kendall", "Lamar", "Omar", "Dwayne",
]

LAST_NAMES = [
    "Williams", "Johnson", "Brown", "Jones", "Davis", "Miller", "Wilson",
    "Moore", "Taylor", "Anderson", "Thomas", "Jackson", "White", "Harris",
    "Martin", "Thompson", "Robinson", "Clark", "Lewis", "Walker", "Hall",
    "Young", "Allen", "King", "Wright", "Scott", "Green", "Baker",
    "Adams", "Nelson", "Carter", "Mitchell", "Perez", "Roberts", "Turner",
    "Phillips", "Campbell", "Parker", "Evans", "Edwards", "Collins",
    "Stewart", "Sanchez", "Morris", "Rogers", "Reed", "Cook", "Morgan",
    "Bell", "Murphy", "Bailey", "Rivera", "Cooper", "Richardson", "Cox",
]

# ~360 D1 team names (subset for generation)
D1_TEAMS = [
    # ACC (current)
    *CURRENT_ACC_TEAMS,
    # Big Ten
    "Michigan", "Ohio St.", "Michigan St.", "Purdue", "Indiana",
    "Illinois", "Iowa", "Wisconsin", "Minnesota", "Nebraska",
    "Northwestern", "Maryland", "Rutgers", "Penn St.",
    "Oregon", "Washington", "UCLA", "USC",
    # SEC
    "Kentucky", "Tennessee", "Auburn", "Alabama", "Arkansas",
    "Mississippi St.", "Ole Miss", "LSU", "Missouri", "South Carolina",
    "Vanderbilt", "Georgia", "Florida", "Texas A&M", "Texas", "Oklahoma",
    # Big 12
    "Kansas", "Baylor", "Texas Tech", "TCU", "Oklahoma St.",
    "Iowa St.", "West Virginia", "Kansas St.", "Cincinnati",
    "Houston", "UCF", "BYU", "Colorado", "Arizona", "Arizona St.", "Utah",
    # Big East
    "UConn", "Villanova", "Creighton", "Marquette", "Xavier",
    "Providence", "St. John's", "Seton Hall", "Butler", "DePaul",
    "Georgetown",
    # Mid-majors (sample)
    "Gonzaga", "San Diego St.", "Memphis", "Saint Mary's", "Dayton",
    "VCU", "Davidson", "Loyola Chicago", "Drake", "Murray St.",
    "Furman", "Charleston", "UAB", "North Texas", "Louisiana Tech",
    "UNLV", "New Mexico", "Nevada", "Boise St.", "Utah St.",
    "Toledo", "Akron", "Kent St.", "Ohio", "Buffalo",
    "Iona", "Fairfield", "Marist", "Siena", "Manhattan",
    "Vermont", "UMBC", "Albany", "Hartford", "Stony Brook",
    "Winthrop", "Hampton", "Radford", "Campbell", "Longwood",
    "Colgate", "Bucknell", "Lehigh", "Navy", "Army",
    "Yale", "Princeton", "Harvard", "Brown", "Cornell",
    "Belmont", "Lipscomb", "North Florida", "Jacksonville St.",
    "Eastern Kentucky", "Morehead St.", "UT Martin", "SE Missouri",
    "Prairie View", "Alcorn St.", "Grambling", "Jackson St.",
    "Weber St.", "Montana", "Northern Colorado", "Portland St.",
    "Sacramento St.", "Idaho St.", "Eastern Washington",
    "Sam Houston", "Tarleton", "Abilene Christian", "Grand Canyon",
    "Seattle", "Utah Valley", "Southern Utah", "Cal Baptist",
    "Chattanooga", "ETSU", "Mercer", "Samford", "UNC Greensboro",
    "Coastal Carolina", "Appalachian St.", "Georgia St.", "Georgia Southern",
    "James Madison", "Marshall", "Old Dominion", "Southern Miss",
    "UTEP", "FIU", "Western Kentucky", "Middle Tennessee",
    "Wichita St.", "Tulane", "South Florida", "East Carolina",
    "Temple", "Tulsa", "FAU", "Rice", "Charlotte", "UTSA",
    "North Alabama", "Kennesaw St.", "Queens", "Lindenwood",
    "Stonehill", "Le Moyne", "Mercyhurst", "St. Thomas",
]

TEAM_TO_CONF = {}  # populated in _assign_conferences()


def _assign_conferences() -> dict[str, str]:
    """Map each team to a conference. Simplified but consistent."""
    mapping = {}
    acc = CURRENT_ACC_TEAMS
    b10 = D1_TEAMS[18:36]
    sec = D1_TEAMS[36:52]
    b12 = D1_TEAMS[52:68]
    be = D1_TEAMS[68:79]
    for t in acc:
        mapping[t] = "ACC"
    for t in b10:
        mapping[t] = "B10"
    for t in sec:
        mapping[t] = "SEC"
    for t in b12:
        mapping[t] = "B12"
    for t in be:
        mapping[t] = "BE"
    # Assign remaining teams to mid-major conferences
    mid_confs = ["WCC", "MWC", "A10", "MVC", "CAA", "MAC", "SBC",
                 "CUSA", "Amer", "SC", "OVC", "Horz", "Sum", "WAC",
                 "BSky", "BW", "Ivy", "Pat", "MAAC", "NEC", "MEAC",
                 "SWAC", "ASun"]
    remaining = [t for t in D1_TEAMS if t not in mapping]
    for i, t in enumerate(remaining):
        mapping[t] = mid_confs[i % len(mid_confs)]
    return mapping


TEAM_TO_CONF = _assign_conferences()


def _generate_player_id(name: str, team: str, season: int) -> str:
    """Deterministic player ID from name+team+season."""
    raw = f"{name}_{team}_{season}"
    return hashlib.md5(raw.encode()).hexdigest()[:10]


def _generate_player_name(rng: np.random.Generator) -> str:
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    return f"{first} {last}"


def _team_quality_factor(team: str) -> float:
    """Higher for power conference / historically strong programs."""
    conf = TEAM_TO_CONF.get(team, "Ind")
    base = {
        "ACC": 0.70, "B10": 0.70, "SEC": 0.72, "B12": 0.68, "BE": 0.65,
        "WCC": 0.50, "MWC": 0.48, "A10": 0.50, "MVC": 0.45,
    }
    conf_factor = base.get(conf, 0.35)
    # Add team-specific bumps for blue bloods
    blue_bloods = {"Duke": 0.12, "North Carolina": 0.11, "Kansas": 0.12,
                   "Kentucky": 0.11, "Gonzaga": 0.10, "UConn": 0.09,
                   "Villanova": 0.08, "Purdue": 0.07, "Houston": 0.07,
                   "Tennessee": 0.07, "Auburn": 0.07, "Alabama": 0.06}
    return conf_factor + blue_bloods.get(team, 0.0)


def generate_team_stats(seasons: list[int] = SEASONS,
                        seed: int = 42) -> pl.DataFrame:
    """
    Generate team-level stats mirroring BartTorvik team ratings.

    Columns: team, conf, barthag, adj_o, adj_d, adj_t, wab, seed,
             year, sos, sos_adj, rec_w, rec_l
    """
    rng = np.random.default_rng(seed)
    rows = []

    for season in seasons:
        for team in D1_TEAMS:
            quality = _team_quality_factor(team)

            # Adjusted offensive efficiency (points per 100 possessions)
            # D1 range: ~85-125, mean ~105
            adj_o = rng.normal(loc=100 + quality * 25, scale=4.5)
            adj_o = np.clip(adj_o, 82, 130)

            # Adjusted defensive efficiency (lower = better)
            adj_d = rng.normal(loc=105 - quality * 18, scale=5.0)
            adj_d = np.clip(adj_d, 82, 120)

            # Adjusted tempo (possessions per 40 min)
            adj_t = rng.normal(loc=67.5, scale=3.5)
            adj_t = np.clip(adj_t, 58, 78)

            # Barthag (probability of beating average D1 team)
            barthag = 1 / (1 + np.exp(-(adj_o - adj_d) / 10))

            # Strength of schedule
            conf = TEAM_TO_CONF.get(team, "Ind")
            conf_sos = {"ACC": 7.5, "B10": 8.0, "SEC": 8.5, "B12": 7.0,
                        "BE": 6.5, "WCC": 2.5, "MWC": 3.0, "A10": 4.0}
            sos = rng.normal(loc=conf_sos.get(conf, 0.0), scale=2.0)
            sos_adj = sos + rng.normal(0, 0.5)

            # Wins above bubble
            wab = (barthag - 0.5) * 30 + rng.normal(0, 2)

            # NCAA tournament seed (None for most teams)
            seed_val = None
            if barthag > 0.85:
                seed_val = int(np.clip(rng.integers(1, 5), 1, 4))
            elif barthag > 0.75:
                seed_val = int(np.clip(rng.integers(4, 9), 4, 8))
            elif barthag > 0.65:
                seed_val = int(np.clip(rng.integers(8, 14), 8, 13))
            elif barthag > 0.55 and rng.random() > 0.5:
                seed_val = int(rng.integers(13, 17))

            # Record
            win_pct = barthag * 0.85 + rng.normal(0, 0.05)
            win_pct = np.clip(win_pct, 0.1, 0.95)
            total_games = rng.integers(28, 36)
            rec_w = int(round(win_pct * total_games))
            rec_l = int(total_games - rec_w)

            rows.append({
                "team": team,
                "conf": conf,
                "barthag": round(float(barthag), 4),
                "adj_o": round(float(adj_o), 1),
                "adj_d": round(float(adj_d), 1),
                "adj_t": round(float(adj_t), 1),
                "wab": round(float(wab), 1),
                "seed": seed_val,
                "year": season,
                "sos": round(float(sos), 2),
                "sos_adj": round(float(sos_adj), 2),
                "rec_w": rec_w,
                "rec_l": rec_l,
            })

    return pl.DataFrame(rows)


def generate_player_stats(team_stats: pl.DataFrame,
                          players_per_team: int = 12,
                          seed: int = 42) -> pl.DataFrame:
    """
    Generate player-level stats mirroring BartTorvik player pages.

    Columns: player, team, conf, g, mpg, ortg, usg, efg, ts_pct, ast_pct,
             to_pct, orb_pct, drb_pct, blk_pct, stl_pct, ftr, porpag,
             bpm, obpm, dbpm, year, pid, class_year, height_in, recruiting_stars
    """
    rng = np.random.default_rng(seed)
    rows = []

    team_season_map = {}
    for row in team_stats.iter_rows(named=True):
        team_season_map[(row["team"], row["year"])] = row

    for (team, year), trow in team_season_map.items():
        team_adj_o = trow["adj_o"]
        team_adj_d = trow["adj_d"]
        conf = trow["conf"]
        quality = _team_quality_factor(team)

        for p in range(players_per_team):
            name = _generate_player_name(rng)
            pid = _generate_player_id(name, team, year)

            # Role hierarchy: first 5 are starters, rest bench
            is_starter = p < 5
            is_star = p < 2

            # Minutes per game
            if is_star:
                mpg = rng.normal(32, 2.5)
            elif is_starter:
                mpg = rng.normal(27, 3.0)
            else:
                mpg = rng.normal(14, 5.0)
            mpg = np.clip(mpg, 3, 40)

            # Games played
            g = int(np.clip(rng.normal(28, 4), 8, 35))

            # Offensive rating (per 100 poss, individual)
            # Correlated with team adj_o but with individual variance
            base_ortg = team_adj_o + rng.normal(0, 8)
            if is_star:
                base_ortg += rng.normal(5, 3)
            ortg = np.clip(base_ortg, 70, 140)

            # Usage rate (% of team possessions used)
            if is_star:
                usg = rng.normal(27, 3)
            elif is_starter:
                usg = rng.normal(20, 3)
            else:
                usg = rng.normal(15, 4)
            usg = np.clip(usg, 8, 40)

            # Effective FG%
            efg = rng.normal(0.50, 0.06)
            efg = np.clip(efg, 0.28, 0.72)

            # True shooting %
            ts_pct = efg + rng.normal(0.04, 0.02)
            ts_pct = np.clip(ts_pct, 0.30, 0.75)

            # Assist rate
            ast_pct = rng.normal(15, 7)
            if p == 0:  # point guard type
                ast_pct += rng.normal(10, 3)
            ast_pct = np.clip(ast_pct, 2, 45)

            # Turnover rate
            to_pct = rng.normal(17, 5)
            to_pct = np.clip(to_pct, 5, 35)

            # Rebound rates
            orb_pct = rng.normal(5, 3)
            drb_pct = rng.normal(14, 5)
            if p >= 3 and p <= 4:  # big man types
                orb_pct += rng.normal(4, 2)
                drb_pct += rng.normal(6, 3)
            orb_pct = np.clip(orb_pct, 0.5, 18)
            drb_pct = np.clip(drb_pct, 3, 30)

            # Block and steal rates
            blk_pct = rng.lognormal(1.0, 0.8)
            blk_pct = np.clip(blk_pct, 0.1, 15)
            stl_pct = rng.normal(2.0, 1.0)
            stl_pct = np.clip(stl_pct, 0.2, 6.0)

            # Free throw rate
            ftr = rng.normal(0.32, 0.12)
            ftr = np.clip(ftr, 0.05, 0.75)

            # Points over replacement per adjusted game
            porpag = (ortg - 100) * (usg / 20) * (mpg / 30) + rng.normal(0, 1.5)

            # Box plus/minus
            obpm = (ortg - 105) / 10 * (usg / 20) + rng.normal(0, 1.5)
            dbpm = -(team_adj_d - 100) / 15 + rng.normal(0, 1.5)
            bpm = obpm + dbpm

            # Class year
            class_year = rng.choice(["Fr", "So", "Jr", "Sr"], p=[0.25, 0.25, 0.25, 0.25])

            # Height (inches)
            height_in = int(rng.normal(77, 3))  # ~6'5" average
            height_in = np.clip(height_in, 69, 87)

            # Recruiting stars (correlated with team quality)
            if quality > 0.7:
                stars = rng.choice([3, 4, 5], p=[0.3, 0.5, 0.2])
            elif quality > 0.5:
                stars = rng.choice([2, 3, 4], p=[0.3, 0.5, 0.2])
            else:
                stars = rng.choice([0, 2, 3], p=[0.4, 0.4, 0.2])

            rows.append({
                "player": name,
                "team": team,
                "conf": conf,
                "g": g,
                "mpg": round(float(mpg), 1),
                "ortg": round(float(ortg), 1),
                "usg": round(float(usg), 1),
                "efg": round(float(efg), 3),
                "ts_pct": round(float(ts_pct), 3),
                "ast_pct": round(float(ast_pct), 1),
                "to_pct": round(float(to_pct), 1),
                "orb_pct": round(float(orb_pct), 1),
                "drb_pct": round(float(drb_pct), 1),
                "blk_pct": round(float(blk_pct), 1),
                "stl_pct": round(float(stl_pct), 1),
                "ftr": round(float(ftr), 3),
                "porpag": round(float(porpag), 2),
                "bpm": round(float(bpm), 1),
                "obpm": round(float(obpm), 1),
                "dbpm": round(float(dbpm), 1),
                "year": year,
                "pid": pid,
                "class_year": class_year,
                "height_in": int(height_in),
                "recruiting_stars": int(stars),
            })

    return pl.DataFrame(rows)


def generate_transfer_portal(player_stats: pl.DataFrame,
                             team_stats: pl.DataFrame,
                             transfer_rate: float = 0.08,
                             seed: int = 42) -> pl.DataFrame:
    """
    Generate realistic transfer portal entries with pre/post stats.

    Logic:
    - ~8% of players transfer each year (realistic post-2021 rate)
    - Transfers are more likely for: bench players, players at weaker programs,
      underclassmen, and players with lower BPM relative to team
    - Post-transfer stats incorporate destination team quality, conference
      adjustment, and individual trajectory noise
    - ~55% of transfers show improvement (realistic rate)

    Columns: player, pid, origin_team, origin_conf, dest_team, dest_conf,
             transfer_season, pre_* (all player stats), post_* (all player stats),
             success (binary target), recruiting_stars
    """
    rng = np.random.default_rng(seed)
    rows = []

    # Build team stats lookup
    team_lookup = {}
    for row in team_stats.iter_rows(named=True):
        team_lookup[(row["team"], row["year"])] = row

    # Process each season (transfers happen between seasons)
    stat_cols = ["g", "mpg", "ortg", "usg", "efg", "ts_pct", "ast_pct",
                 "to_pct", "orb_pct", "drb_pct", "blk_pct", "stl_pct",
                 "ftr", "porpag", "bpm", "obpm", "dbpm"]

    for season_idx in range(len(SEASONS) - 1):
        origin_season = SEASONS[season_idx]
        dest_season = SEASONS[season_idx + 1]

        # Get players from origin season
        season_players = player_stats.filter(pl.col("year") == origin_season)

        for row in season_players.iter_rows(named=True):
            # Transfer probability — higher for bench, weaker teams, underclassmen
            p_transfer = transfer_rate
            if row["mpg"] < 15:
                p_transfer *= 1.8
            if row["bpm"] < -2:
                p_transfer *= 1.5
            if row["class_year"] in ("Fr", "So"):
                p_transfer *= 1.3
            if row["class_year"] == "Sr":
                p_transfer *= 0.3  # seniors rarely transfer
            quality = _team_quality_factor(row["team"])
            if quality < 0.5:
                p_transfer *= 1.4

            if rng.random() > p_transfer:
                continue

            # Choose destination team (different from origin)
            origin_team = row["team"]
            available_teams = [t for t in D1_TEAMS if t != origin_team]

            # Weight toward: teams in better/similar conferences, upward mobility
            weights = []
            for t in available_teams:
                dest_q = _team_quality_factor(t)
                # Slight preference for better programs
                w = 1.0 + max(0, dest_q - quality) * 3
                weights.append(w)
            weights = np.array(weights)
            weights /= weights.sum()

            dest_team = rng.choice(available_teams, p=weights)
            dest_conf = TEAM_TO_CONF.get(dest_team, "Ind")

            # Get destination team stats
            dest_team_stats = team_lookup.get((dest_team, dest_season))
            origin_team_stats = team_lookup.get((origin_team, origin_season))

            if dest_team_stats is None or origin_team_stats is None:
                continue

            # Generate post-transfer stats
            # Key factors: destination team quality, role change, conference jump
            dest_quality = _team_quality_factor(dest_team)
            quality_delta = dest_quality - quality

            # Conference strength adjustment
            conf_adj = (dest_team_stats["adj_o"] - origin_team_stats["adj_o"]) / 10

            # Post-transfer ORtg: base is pre-transfer, adjusted for context
            # Moving to a better team often means lower usage but system boost
            # Moving to a worse team often means higher usage but less support
            post_ortg = row["ortg"] + conf_adj * 2 + rng.normal(0, 6)

            # Usage tends to change inversely with destination quality
            post_usg = row["usg"] - quality_delta * 8 + rng.normal(0, 3)
            post_usg = np.clip(post_usg, 8, 38)

            # Minutes change
            if row["mpg"] < 15:
                # Bench players often transfer for playing time
                post_mpg = row["mpg"] + rng.normal(8, 4)
            else:
                post_mpg = row["mpg"] + rng.normal(-2 * quality_delta * 10, 4)
            post_mpg = np.clip(post_mpg, 5, 38)

            # Other stats with realistic noise
            post_efg = row["efg"] + rng.normal(0, 0.03)
            post_ts = row["ts_pct"] + rng.normal(0, 0.03)
            post_ast = row["ast_pct"] + rng.normal(0, 3)
            post_to = row["to_pct"] + rng.normal(0, 2.5)
            post_orb = row["orb_pct"] + rng.normal(0, 1.5)
            post_drb = row["drb_pct"] + rng.normal(0, 2)
            post_blk = row["blk_pct"] + rng.normal(0, 0.8)
            post_stl = row["stl_pct"] + rng.normal(0, 0.5)
            post_ftr = row["ftr"] + rng.normal(0, 0.05)

            # BPM and PORPAG recalculated
            post_obpm = (post_ortg - 105) / 10 * (post_usg / 20) + rng.normal(0, 1)
            post_dbpm = -(dest_team_stats["adj_d"] - 100) / 15 + rng.normal(0, 1)
            post_bpm = post_obpm + post_dbpm
            post_porpag = ((post_ortg - 100) * (post_usg / 20) *
                           (post_mpg / 30) + rng.normal(0, 1))
            post_g = int(np.clip(rng.normal(27, 5), 5, 35))

            # Clip everything
            post_ortg = np.clip(post_ortg, 70, 140)
            post_efg = np.clip(post_efg, 0.28, 0.72)
            post_ts = np.clip(post_ts, 0.30, 0.75)
            post_ast = np.clip(post_ast, 2, 45)
            post_to = np.clip(post_to, 5, 35)
            post_orb = np.clip(post_orb, 0.5, 18)
            post_drb = np.clip(post_drb, 3, 30)
            post_blk = np.clip(post_blk, 0.1, 15)
            post_stl = np.clip(post_stl, 0.2, 6)
            post_ftr = np.clip(post_ftr, 0.05, 0.75)

            # --- Target variable ---
            # Per-40 adjusted ORtg: ortg * (40 / mpg) normalized
            pre_per40_ortg = row["ortg"]   # already per-100-poss, use directly
            post_per40_ortg = post_ortg
            success = int(post_per40_ortg >= pre_per40_ortg)

            transfer_row = {
                "player": row["player"],
                "pid": row["pid"],
                "origin_team": origin_team,
                "origin_conf": row["conf"],
                "dest_team": dest_team,
                "dest_conf": dest_conf,
                "transfer_season": dest_season,
                "recruiting_stars": row["recruiting_stars"],
                "class_year": row["class_year"],
                "height_in": row["height_in"],
                # Pre-transfer stats
                "pre_g": row["g"],
                "pre_mpg": row["mpg"],
                "pre_ortg": row["ortg"],
                "pre_usg": row["usg"],
                "pre_efg": row["efg"],
                "pre_ts_pct": row["ts_pct"],
                "pre_ast_pct": row["ast_pct"],
                "pre_to_pct": row["to_pct"],
                "pre_orb_pct": row["orb_pct"],
                "pre_drb_pct": row["drb_pct"],
                "pre_blk_pct": row["blk_pct"],
                "pre_stl_pct": row["stl_pct"],
                "pre_ftr": row["ftr"],
                "pre_porpag": row["porpag"],
                "pre_bpm": row["bpm"],
                "pre_obpm": row["obpm"],
                "pre_dbpm": row["dbpm"],
                # Post-transfer stats
                "post_g": post_g,
                "post_mpg": round(float(post_mpg), 1),
                "post_ortg": round(float(post_ortg), 1),
                "post_usg": round(float(post_usg), 1),
                "post_efg": round(float(post_efg), 3),
                "post_ts_pct": round(float(post_ts), 3),
                "post_ast_pct": round(float(post_ast), 1),
                "post_to_pct": round(float(post_to), 1),
                "post_orb_pct": round(float(post_orb), 1),
                "post_drb_pct": round(float(post_drb), 1),
                "post_blk_pct": round(float(post_blk), 1),
                "post_stl_pct": round(float(post_stl), 1),
                "post_ftr": round(float(post_ftr), 3),
                "post_porpag": round(float(post_porpag), 2),
                "post_bpm": round(float(post_bpm), 1),
                "post_obpm": round(float(post_obpm), 1),
                "post_dbpm": round(float(post_dbpm), 1),
                # Origin/dest team context
                "origin_adj_o": origin_team_stats["adj_o"],
                "origin_adj_d": origin_team_stats["adj_d"],
                "origin_adj_t": origin_team_stats["adj_t"],
                "origin_barthag": origin_team_stats["barthag"],
                "dest_adj_o": dest_team_stats["adj_o"],
                "dest_adj_d": dest_team_stats["adj_d"],
                "dest_adj_t": dest_team_stats["adj_t"],
                "dest_barthag": dest_team_stats["barthag"],
                # Target
                "success": success,
            }
            rows.append(transfer_row)

    return pl.DataFrame(rows)


def generate_coaching_data(seed: int = 42) -> pl.DataFrame:
    """Generate coaching tenure data for ranking model."""
    rng = np.random.default_rng(seed)
    rows = []
    for team in D1_TEAMS:
        tenure = int(np.clip(rng.exponential(6), 1, 30))
        for season in SEASONS:
            # Small chance of coaching change each year
            if rng.random() < 0.08 and season > SEASONS[0]:
                tenure = 1
            else:
                tenure += 1
            rows.append({
                "team": team,
                "year": season,
                "coaching_tenure_years": tenure,
            })
    return pl.DataFrame(rows)


def generate_returning_production(player_stats: pl.DataFrame,
                                  seed: int = 42) -> pl.DataFrame:
    """
    Calculate returning production % for each team-season.
    Returning production = % of prior season's minutes that return.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for team in D1_TEAMS:
        for season in SEASONS:
            # Realistic returning production: mean ~55%, range 20-85%
            quality = _team_quality_factor(team)
            # Better teams lose more to NBA/transfers
            base_ret = 0.60 - quality * 0.15
            ret_pct = np.clip(rng.normal(base_ret, 0.12), 0.15, 0.90)

            # Incoming transfer talent (composite rating)
            # Higher for teams actively using portal
            n_transfers_in = int(np.clip(rng.poisson(2.5), 0, 8))
            if quality > 0.6:
                avg_star = rng.normal(3.2, 0.5)
            else:
                avg_star = rng.normal(2.3, 0.6)
            incoming_transfer_composite = n_transfers_in * np.clip(avg_star, 0, 5)

            rows.append({
                "team": team,
                "year": season,
                "returning_production_pct": round(float(ret_pct), 3),
                "n_transfers_in": n_transfers_in,
                "incoming_transfer_composite": round(float(incoming_transfer_composite), 2),
            })

    return pl.DataFrame(rows)


def generate_all_data(seed: int = 42) -> dict[str, pl.DataFrame]:
    """Generate all datasets and return as dict of DataFrames."""
    print("Generating team stats...")
    team_stats = generate_team_stats(seed=seed)

    print("Generating player stats...")
    player_stats = generate_player_stats(team_stats, seed=seed)

    print("Generating transfer portal data...")
    transfers = generate_transfer_portal(player_stats, team_stats, seed=seed)

    print("Generating coaching data...")
    coaching = generate_coaching_data(seed=seed)

    print("Generating returning production data...")
    ret_prod = generate_returning_production(player_stats, seed=seed)

    return {
        "team_stats": team_stats,
        "player_stats": player_stats,
        "transfers": transfers,
        "coaching": coaching,
        "returning_production": ret_prod,
    }


if __name__ == "__main__":
    data = generate_all_data()
    for name, df in data.items():
        path = f"data/raw/{name}.parquet"
        df.write_parquet(path)
        print(f"  {name}: {df.shape} -> {path}")
    print("\nDone. All raw data written to data/raw/")
