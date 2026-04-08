-- ============================================================================
-- NCAA Basketball Transfer Portal — Analytical SQL Queries (DuckDB)
-- ============================================================================
-- Run via: duckdb < sql/queries.sql
-- Or programmatically via the Python DuckDB interface.
--
-- These queries demonstrate:
--   - Window functions (ROW_NUMBER, LAG, RANK, AVG OVER)
--   - CTEs (Common Table Expressions)
--   - Conference-adjusted statistics
--   - Pre/post transfer comparisons
--   - Team roster composition analysis
-- ============================================================================

-- Load parquet files into DuckDB views
CREATE OR REPLACE VIEW transfers AS
    SELECT * FROM read_parquet('data/raw/transfers.parquet');

CREATE OR REPLACE VIEW player_stats AS
    SELECT * FROM read_parquet('data/raw/player_stats.parquet');

CREATE OR REPLACE VIEW team_stats AS
    SELECT * FROM read_parquet('data/raw/team_stats.parquet');

CREATE OR REPLACE VIEW coaching AS
    SELECT * FROM read_parquet('data/raw/coaching.parquet');

CREATE OR REPLACE VIEW returning_production AS
    SELECT * FROM read_parquet('data/raw/returning_production.parquet');


-- ============================================================================
-- Query 1: Transfer Success Rate by Conference Jump Direction
-- Uses: CTE, CASE WHEN, GROUP BY aggregation
-- ============================================================================

WITH conf_jumps AS (
    SELECT
        *,
        CASE
            WHEN origin_conf IN ('ACC','B10','SEC','B12','BE')
                 AND dest_conf IN ('ACC','B10','SEC','B12','BE') THEN 'Power → Power'
            WHEN origin_conf IN ('ACC','B10','SEC','B12','BE')
                 AND dest_conf NOT IN ('ACC','B10','SEC','B12','BE') THEN 'Power → Mid'
            WHEN origin_conf NOT IN ('ACC','B10','SEC','B12','BE')
                 AND dest_conf IN ('ACC','B10','SEC','B12','BE') THEN 'Mid → Power'
            ELSE 'Mid → Mid'
        END AS jump_type
    FROM transfers
)
SELECT
    jump_type,
    COUNT(*) AS n_transfers,
    ROUND(AVG(success), 3) AS success_rate,
    ROUND(AVG(post_ortg - pre_ortg), 2) AS avg_ortg_change,
    ROUND(AVG(post_bpm - pre_bpm), 2) AS avg_bpm_change
FROM conf_jumps
GROUP BY jump_type
ORDER BY success_rate DESC;


-- ============================================================================
-- Query 2: Top Transfer Destinations by Incoming Talent Rating
-- Uses: Window function (RANK), aggregate with HAVING
-- ============================================================================

WITH dest_summary AS (
    SELECT
        dest_team,
        dest_conf,
        transfer_season,
        COUNT(*) AS n_incoming,
        ROUND(AVG(recruiting_stars), 2) AS avg_stars_incoming,
        ROUND(AVG(pre_bpm), 2) AS avg_pre_bpm_incoming,
        ROUND(AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END), 3) AS success_rate
    FROM transfers
    GROUP BY dest_team, dest_conf, transfer_season
    HAVING COUNT(*) >= 2
)
SELECT
    *,
    RANK() OVER (PARTITION BY transfer_season ORDER BY avg_pre_bpm_incoming DESC)
        AS talent_rank
FROM dest_summary
QUALIFY talent_rank <= 10
ORDER BY transfer_season, talent_rank;


-- ============================================================================
-- Query 3: Pre vs Post Transfer Performance by Class Year
-- Uses: Window function (AVG OVER), conditional aggregation
-- ============================================================================

SELECT
    class_year,
    COUNT(*) AS n,
    ROUND(AVG(pre_ortg), 1) AS avg_pre_ortg,
    ROUND(AVG(post_ortg), 1) AS avg_post_ortg,
    ROUND(AVG(post_ortg - pre_ortg), 2) AS avg_ortg_delta,
    ROUND(AVG(pre_mpg), 1) AS avg_pre_mpg,
    ROUND(AVG(post_mpg), 1) AS avg_post_mpg,
    ROUND(AVG(post_mpg - pre_mpg), 1) AS avg_mpg_delta,
    ROUND(AVG(success), 3) AS success_rate,
    -- Compare to overall average using window function
    ROUND(AVG(success) - AVG(AVG(success)) OVER (), 3) AS success_rate_vs_avg
FROM transfers
GROUP BY class_year
ORDER BY
    CASE class_year WHEN 'Fr' THEN 1 WHEN 'So' THEN 2
         WHEN 'Jr' THEN 3 WHEN 'Sr' THEN 4 END;


-- ============================================================================
-- Query 4: Conference-Adjusted Player Efficiency Rankings
-- Uses: Window function (RANK, AVG OVER PARTITION), CTE
-- ============================================================================

WITH conf_baselines AS (
    SELECT
        conf,
        year,
        AVG(ortg) AS conf_avg_ortg,
        AVG(bpm) AS conf_avg_bpm
    FROM player_stats
    WHERE mpg >= 15
    GROUP BY conf, year
),
adjusted AS (
    SELECT
        p.player,
        p.team,
        p.conf,
        p.year,
        p.ortg,
        p.bpm,
        p.mpg,
        p.usg,
        ROUND(p.ortg - c.conf_avg_ortg, 2) AS conf_adj_ortg,
        ROUND(p.bpm - c.conf_avg_bpm, 2) AS conf_adj_bpm
    FROM player_stats p
    JOIN conf_baselines c ON p.conf = c.conf AND p.year = c.year
    WHERE p.mpg >= 20
)
SELECT
    *,
    RANK() OVER (PARTITION BY conf, year ORDER BY conf_adj_bpm DESC) AS conf_rank
FROM adjusted
QUALIFY conf_rank <= 5
ORDER BY year, conf, conf_rank;


-- ============================================================================
-- Query 5: Rolling Team Efficiency Trend (3-Year Window)
-- Uses: Window function (AVG OVER ROWS BETWEEN), LAG
-- ============================================================================

SELECT
    team,
    conf,
    year,
    ROUND(adj_o - adj_d, 2) AS eff_margin,
    ROUND(LAG(adj_o - adj_d) OVER (PARTITION BY team ORDER BY year), 2)
        AS prev_eff_margin,
    ROUND(
        AVG(adj_o - adj_d) OVER (
            PARTITION BY team ORDER BY year
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ), 2
    ) AS rolling_3yr_eff_margin,
    ROUND(
        (adj_o - adj_d) - LAG(adj_o - adj_d) OVER (PARTITION BY team ORDER BY year), 2
    ) AS yoy_change
FROM team_stats
WHERE team IN ('Duke', 'North Carolina', 'Virginia', 'Louisville', 'SMU',
               'Florida St.', 'Clemson', 'Pittsburgh')
ORDER BY team, year;


-- ============================================================================
-- Query 6: Team Roster Composition via Transfer Portal
-- Uses: Subquery, LEFT JOIN, COALESCE
-- ============================================================================

WITH team_transfers AS (
    SELECT
        dest_team AS team,
        transfer_season AS year,
        COUNT(*) AS n_transfers_in,
        ROUND(AVG(recruiting_stars), 2) AS avg_transfer_stars,
        ROUND(AVG(pre_bpm), 2) AS avg_transfer_pre_bpm
    FROM transfers
    GROUP BY dest_team, transfer_season
),
team_departures AS (
    SELECT
        origin_team AS team,
        transfer_season AS year,
        COUNT(*) AS n_transfers_out
    FROM transfers
    GROUP BY origin_team, transfer_season
)
SELECT
    t.team,
    t.conf,
    t.year,
    t.barthag,
    ROUND(t.adj_o - t.adj_d, 1) AS eff_margin,
    COALESCE(ti.n_transfers_in, 0) AS n_in,
    COALESCE(td.n_transfers_out, 0) AS n_out,
    COALESCE(ti.n_transfers_in, 0) - COALESCE(td.n_transfers_out, 0) AS net_transfers,
    ti.avg_transfer_stars,
    ti.avg_transfer_pre_bpm
FROM team_stats t
LEFT JOIN team_transfers ti ON t.team = ti.team AND t.year = ti.year
LEFT JOIN team_departures td ON t.team = td.team AND t.year = td.year
WHERE t.team IN ('Duke', 'North Carolina', 'Kentucky', 'Kansas', 'Gonzaga')
ORDER BY t.team, t.year;


-- ============================================================================
-- Query 7: Transfer Outcome Distribution by Destination Team Quality Tier
-- Uses: NTILE window function, percentile-based buckets
-- ============================================================================

WITH team_tiers AS (
    SELECT
        team,
        year,
        barthag,
        NTILE(4) OVER (PARTITION BY year ORDER BY barthag DESC) AS quality_tier
    FROM team_stats
)
SELECT
    CASE quality_tier
        WHEN 1 THEN 'Tier 1 (Top 25%)'
        WHEN 2 THEN 'Tier 2 (50-75%)'
        WHEN 3 THEN 'Tier 3 (25-50%)'
        WHEN 4 THEN 'Tier 4 (Bottom 25%)'
    END AS dest_tier,
    COUNT(*) AS n_transfers,
    ROUND(AVG(success), 3) AS success_rate,
    ROUND(AVG(post_ortg - pre_ortg), 2) AS avg_ortg_change,
    ROUND(AVG(post_mpg - pre_mpg), 1) AS avg_mpg_change,
    ROUND(AVG(post_usg - pre_usg), 1) AS avg_usg_change
FROM transfers tr
JOIN team_tiers tt ON tr.dest_team = tt.team AND tr.transfer_season = tt.year
GROUP BY quality_tier
ORDER BY quality_tier;


-- ============================================================================
-- Query 8: Most Impactful Transfers (Per-Season Top 10 by BPM Improvement)
-- Uses: ROW_NUMBER, computed column, QUALIFY
-- ============================================================================

SELECT
    player,
    origin_team,
    origin_conf,
    dest_team,
    dest_conf,
    transfer_season,
    ROUND(pre_bpm, 1) AS pre_bpm,
    ROUND(post_bpm, 1) AS post_bpm,
    ROUND(post_bpm - pre_bpm, 2) AS bpm_improvement,
    ROUND(pre_mpg, 1) AS pre_mpg,
    ROUND(post_mpg, 1) AS post_mpg,
    ROW_NUMBER() OVER (
        PARTITION BY transfer_season
        ORDER BY (post_bpm - pre_bpm) DESC
    ) AS improvement_rank
FROM transfers
WHERE pre_mpg >= 10 AND post_mpg >= 10  -- Minimum playing time filter
QUALIFY improvement_rank <= 10
ORDER BY transfer_season, improvement_rank;


-- ============================================================================
-- Query 9: Coaching Tenure vs Team Performance & Portal Activity
-- Uses: Multi-table JOIN, CASE bucketing, GROUP BY
-- ============================================================================

SELECT
    CASE
        WHEN c.coaching_tenure_years <= 2 THEN '1-2 yrs (New)'
        WHEN c.coaching_tenure_years <= 5 THEN '3-5 yrs (Establishing)'
        WHEN c.coaching_tenure_years <= 10 THEN '6-10 yrs (Established)'
        ELSE '10+ yrs (Veteran)'
    END AS tenure_bucket,
    COUNT(DISTINCT t.team || t.year::VARCHAR) AS n_team_seasons,
    ROUND(AVG(t.adj_o - t.adj_d), 2) AS avg_eff_margin,
    ROUND(AVG(t.barthag), 3) AS avg_barthag,
    ROUND(AVG(rp.n_transfers_in), 1) AS avg_transfers_in,
    ROUND(AVG(rp.returning_production_pct), 3) AS avg_returning_pct
FROM team_stats t
JOIN coaching c ON t.team = c.team AND t.year = c.year
LEFT JOIN returning_production rp ON t.team = rp.team AND t.year = rp.year
WHERE t.conf IN ('ACC', 'B10', 'SEC', 'B12', 'BE')
GROUP BY tenure_bucket
ORDER BY avg_eff_margin DESC;


-- ============================================================================
-- Query 10: Season-over-Season Transfer Volume Trends
-- Uses: LAG window function, percentage change calculation
-- ============================================================================

WITH yearly_counts AS (
    SELECT
        transfer_season,
        COUNT(*) AS total_transfers,
        SUM(CASE WHEN dest_conf IN ('ACC','B10','SEC','B12','BE') THEN 1 ELSE 0 END)
            AS power_conf_destinations,
        ROUND(AVG(recruiting_stars), 2) AS avg_stars,
        ROUND(AVG(success), 3) AS overall_success_rate
    FROM transfers
    GROUP BY transfer_season
)
SELECT
    transfer_season,
    total_transfers,
    LAG(total_transfers) OVER (ORDER BY transfer_season) AS prev_year_transfers,
    ROUND(
        100.0 * (total_transfers - LAG(total_transfers) OVER (ORDER BY transfer_season))
        / NULLIF(LAG(total_transfers) OVER (ORDER BY transfer_season), 0), 1
    ) AS yoy_pct_change,
    power_conf_destinations,
    ROUND(100.0 * power_conf_destinations / total_transfers, 1) AS pct_to_power,
    avg_stars,
    overall_success_rate
FROM yearly_counts
ORDER BY transfer_season;
