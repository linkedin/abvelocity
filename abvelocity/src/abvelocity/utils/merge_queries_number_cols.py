# BSD 2-CLAUSE LICENSE
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# Original author: Reza Hosseini

from typing import List, Optional, Union


def merge_queries_number_cols(
    queries: List[str],
    on_cols: List[str],
    common_cols: List[str],
    how: str = "inner",
    add_tuple: bool = False,
    drop_numbered_cols: bool = False,
    nan_replacements: Optional[List[Union[str, int, float, None]]] = None,
) -> str:
    """
    Generate SQL to merge multiple queries on join keys, with numbered common columns and optional tuple column.

    Args:
        queries: List of SQL queries or table names as strings.
        on_cols: List of columns to join on.
        common_cols: List of common columns to merge with numbered suffixes.
        how: Join type (inner, left, right, full, outer), case-insensitive.
        add_tuple: If True, add a tuple column combining common columns.
        drop_numbered_cols: If True, drop numbered columns in final select, keep only tuple alias.
        nan_replacements: Optional list of replacements for NULL values in common columns,
            applied to each common column in order. If provided, must match len(common_cols).
            Strings are used as SQL string literals (wrapped in single quotes).

    Returns:
        SQL query string performing the join and column merging.

    Raises:
        ValueError: If no queries are provided or if len(nan_replacements) != len(common_cols) when not None.
    """
    n = len(queries)
    if n == 0:
        raise ValueError("At least one query must be provided")

    if nan_replacements is not None and len(nan_replacements) != len(common_cols):
        raise ValueError(
            f"Length of nan_replacements ({len(nan_replacements)}) must match length of common_cols ({len(common_cols)})"
        )

    join_type_map = {
        "inner": "INNER",
        "left": "LEFT OUTER",
        "right": "RIGHT OUTER",
        "full": "FULL OUTER",
        "outer": "FULL OUTER",
    }

    how_lower = how.lower()
    how_sql = join_type_map[how_lower]

    def col_with_nan_replacement(src_alias: str, col: str, col_idx: int) -> str:
        if nan_replacements is not None:
            val = nan_replacements[col_idx]
            if val is not None:
                if isinstance(val, str):
                    val_str = f"'{val}'"
                else:
                    val_str = str(val)
                return f"COALESCE({src_alias}.{col}, {val_str})"
        return f"{src_alias}.{col}"

    # Build CTEs: one per query with selected columns, aliasing common_cols with numbered suffixes
    cte_clauses = []
    for i, query in enumerate(queries):
        src_alias = f"src_{i}"
        select_cols = []
        for c in on_cols:
            select_cols.append(f"{src_alias}.{c}")
        for j, c in enumerate(common_cols):
            expr = col_with_nan_replacement(src_alias, c, j)
            alias = f"{c}_{i+1}"
            select_cols.append(f"{expr} AS {alias}")
        cte_sql = f"  cte_{i} AS (SELECT {', '.join(select_cols)} FROM ({query}) AS {src_alias})"
        cte_clauses.append(cte_sql)

    # Select all numbered common cols from all ctes
    numbered_cols = []
    for j, c in enumerate(common_cols):
        for i in range(n):
            col_expr = col_with_nan_replacement(f"cte_{i}", f"{c}_{i+1}", j)
            numbered_cols.append(f"{col_expr} AS {c}_{i+1}")

    # Compose join of all CTEs
    if n == 1:
        on_cols_select = [f"cte_0.{col}" for col in on_cols]
        select_cols = on_cols_select + numbered_cols
        initial_merged_data_sql = f"SELECT {', '.join(select_cols)} FROM cte_0"
    else:
        join_expr = "cte_0"
        for i in range(1, n):
            on_condition = " AND ".join([f"cte_{i-1}.{col} = cte_{i}.{col}" for col in on_cols])
            join_expr = f"({join_expr} {how_sql} JOIN cte_{i} ON {on_condition})"

        if how_sql == "FULL OUTER":
            on_cols_select = [
                f"COALESCE({', '.join([f'cte_{j}.{col}' for j in range(n)])}) AS {col}"
                for col in on_cols
            ]
        else:
            on_cols_select = [f"cte_0.{col}" for col in on_cols]

        select_cols = on_cols_select + numbered_cols
        initial_merged_data_sql = f"SELECT {', '.join(select_cols)} FROM {join_expr}"

    # Build final SELECT clause
    final_select_cols = []
    if n == 1 or how_sql != "FULL OUTER":
        for c in on_cols:
            final_select_cols.append(f"_merged_output.{c}")
    else:
        for c in on_cols:
            final_select_cols.append(c)

    if add_tuple:
        for c in common_cols:
            tuple_cols = ", ".join([f"_merged_output.{c}_{i+1}" for i in range(n)])
            final_select_cols.append(f"ROW({tuple_cols}) AS {c}")

        if not drop_numbered_cols:
            for j, c in enumerate(common_cols):
                for i in range(n):
                    col_expr = col_with_nan_replacement("_merged_output", f"{c}_{i+1}", j)
                    final_select_cols.append(f"{col_expr} AS {c}_{i+1}")
    else:
        for j, c in enumerate(common_cols):
            for i in range(n):
                col_expr = col_with_nan_replacement("_merged_output", f"{c}_{i+1}", j)
                final_select_cols.append(f"{col_expr} AS {c}_{i+1}")

    final_select_sql = (
        f"SELECT {', '.join(final_select_cols)} FROM _initial_merged_data AS _merged_output"
    )

    query_sql = (
        "WITH\n"
        + ",\n".join(cte_clauses)
        + ",\n  _initial_merged_data AS (\n"
        + initial_merged_data_sql
        + "\n)\n"
        + final_select_sql
    )

    return query_sql
