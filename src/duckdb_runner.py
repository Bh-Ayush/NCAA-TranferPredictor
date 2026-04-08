"""
Execute analytical SQL queries against the dataset using DuckDB.
Demonstrates DuckDB integration with parquet files.
"""

import duckdb
from pathlib import Path


def run_queries(sql_file: str = "sql/queries.sql", verbose: bool = True) -> dict:
    """
    Execute all SQL queries in the queries file.
    Returns results as a dict of query_name -> DataFrame.
    """
    con = duckdb.connect(database=":memory:")

    sql_text = Path(sql_file).read_text()
    statements = [s.strip() for s in sql_text.split(";") if s.strip()]

    results = {}
    query_num = 0

    for stmt in statements:
        # Skip comments-only blocks
        clean = "\n".join(
            line for line in stmt.split("\n")
            if not line.strip().startswith("--")
        ).strip()

        if not clean:
            continue

        try:
            result = con.execute(clean)

            # CREATE VIEW statements don't return data
            if clean.upper().startswith("CREATE"):
                if verbose:
                    print(f"  [OK] View created")
                continue

            df = result.fetchdf()
            query_num += 1
            results[f"query_{query_num}"] = df

            if verbose:
                print(f"\n  === Query {query_num} ({len(df)} rows) ===")
                print(df.to_string(max_rows=15, max_cols=10))

        except Exception as e:
            if verbose:
                print(f"  [ERROR] {e}")
                print(f"  Statement: {clean[:100]}...")

    con.close()
    return results


if __name__ == "__main__":
    print("Running SQL analytical queries via DuckDB...\n")
    results = run_queries()
    print(f"\n\nExecuted {len(results)} queries successfully.")
