# BSD 2-CLAUSE LICENSE

# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:

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

from typing import Optional

import pandas as pd
from abvelocity.core.get_data.data_container import DataContainer
from abvelocity.core.param.constants import (
    CATEG_NAN_VALUE,
    CONTROL_LABEL,
    TRIGGER_STATE_COL,
    TRIGGER_STATE_COUNT_COL,
    TRIGGER_STATE_OVERALL_COL,
    TRIGGER_STATE_PERCENT_COL,
    VARIANT_COL,
    VARIANT_COUNT_COL,
    VARIANT_OVER_TRIGGERED_PERCENT_COL,
    VARIANT_PERCENT_COL,
)
from abvelocity.core.param.derived_expt_stats import DerivedExptStats
from abvelocity.core.param.io_param import IOParam
from abvelocity.core.param.launch import Launch
from abvelocity.core.param.variant import TriggerState, Variant, variant_to_trigger_state


def calc_variant_count_df(
    dc: DataContainer,
    variant_col: str = VARIANT_COL,
    io_param: Optional[IOParam] = None,
) -> pd.DataFrame:
    """
    This function computes the variant count dataframe for a multi-experiment (could be univariate as well).
    Note that this function assumes that each unit of the experiment appears on only one row of the input dataframe.
    It is generally OK, if this assumption is violated in a small percentage of the data.
    However, we delibrately do not pass an expt_untit_col to this function for simplicity and
    deduplication is done by the caller if needed.

    Args:
        dc: The DataContainer containing the experiment data.
            The data includes `variant_col` column.
        variant_col: The variant column (tuples)
        io_param: Optional IOParam instance, used to extract the Cursor instance
            if the data is in a SQL table/query.

    Returns:
        Computed variant count dataframe.
        Columns:

            - variant_col
            - VARIANT_COUNT_COL

    Raises:
        ValueError: If a SQL source is provided without a Cursor inside io_param.
    """

    cursor = io_param.cursor if io_param is not None else None

    if dc.table_name is None and dc.query is None and dc.pandas_df is None:
        raise ValueError(
            "In `calc_variant_count_df`, the DataContainer has no data source. " "Please provide either a table_name, query, or pandas_df in the DataContainer."
        )

    is_sql_source = (dc.table_name is not None or dc.query is not None) and dc.pandas_df is None

    if is_sql_source:
        # Case 1: Data is sourced from SQL (table_name or query)
        if cursor is None:
            raise ValueError(
                f"A SQL table name or query is provided in the DataContainer: {dc}, "
                "but no 'cursor' instance was supplied (via io_param) to perform the initial aggregation."
            )

        # If `.table_name` we will use that to get the data
        # If not we will use the `.query` provided in dc.
        if dc.table_name is not None:
            sql_source = dc.table_name
            print(f"\n***: In `calc_variant_count_df`, " f"using dc.table_name: {dc.table_name} for computation.")
        elif dc.query is not None:
            sql_source = f"({dc.query})"
            print(f"\n***: In `calc_variant_count_df`, " f"using dc.query: {dc.query} as subquery for computation.")
        else:
            raise ValueError("In `calc_variant_count_df`, both dc.table_name and dc.query are None. " f"dc: {dc}")

        # Construct the SQL aggregation query
        # Note: VARIANT_COUNT_COL is the target column name for the count
        aggregation_query = f"SELECT {variant_col}, COUNT(*) AS {VARIANT_COUNT_COL} " f"FROM {sql_source} " f"GROUP BY 1"

        # Execute the SQL query and retrieve the results directly as the variant count DataFrame
        result = cursor.get_df(aggregation_query)
        variant_count_df = result.df
    elif dc.pandas_df is not None:
        # Case 2: Data is sourced from a Pandas DataFrame (original path)
        variant_count_df = dc.pandas_df.groupby(variant_col, as_index=False).size().rename(columns={"size": VARIANT_COUNT_COL})
    else:
        # Case 3: DataContainer is empty
        raise ValueError("DataContainer has no data (no pandas_df, table_name, or query set).")

    # Make sure variant_col is tuple (Trino returns NamedRowTuple, DuckDB returns dict)
    variant_count_df[variant_col] = [tuple(item.values()) if isinstance(item, dict) else tuple(item) for item in variant_count_df[variant_col]]

    print(f"\n***: In `calc_variant_count_df`, variant_count_df:\n{variant_count_df.head(3)}")
    print(f"\n***: In `calc_variant_count_df`, variant_count_df.dtypes:\n{variant_count_df.dtypes}")

    print(f"\n***: In `calc_variant_count_df`, df[{variant_col}].value_counts():\n" f"{variant_count_df[variant_col].value_counts()}")

    return variant_count_df


def get_expt_stats_vcd(
    variant_count_df: pd.DataFrame,
    variant_col: str = VARIANT_COL,
    trigger_state_col: str = TRIGGER_STATE_COL,
) -> DerivedExptStats:
    """
    This function computes the derived statistics for a multi-experiment (could be univariate as well).
    Note that this function assumes that each unit of the experiment appears on only one row of the input dataframe.
    It is generally OK, if this assumption is violated in a small percentage of the data.
    However, we delibrately do not pass an expt_untit_col to this function for simplicity and
    deduplication is done by the caller if needed.

    Args:

        variant_count_df: Computed variant count dataframe with columns:

            - variant_col
            - VARIANT_COUNT_COL
        variant_col: The variant column name.
        trigger_state_col: The trigger state column name.

    Returns:
        DerivedExptStats: The derived statistics for the experiment.

    """
    print(f"\n***: In `get_expt_stats_vcd`, variant_count_df[{variant_col}].value_counts():\n" f"{variant_count_df[variant_col].value_counts()}")
    # Perform the aggregation using Pandas

    variant_count_df[trigger_state_col] = variant_count_df[variant_col].map(lambda v: variant_to_trigger_state(Variant(value=v)).value)
    variant_count_df[TRIGGER_STATE_OVERALL_COL] = variant_count_df[variant_col].map(lambda v: variant_to_trigger_state(Variant(value=v)).overall_value)

    trigger_state_count_df = variant_count_df.groupby(trigger_state_col).agg(**{TRIGGER_STATE_COUNT_COL: (VARIANT_COUNT_COL, "sum")}).reset_index()

    variant_values = variant_count_df[variant_col].unique()
    variants = [Variant(value=v) for v in variant_values]
    trigger_state_values = variant_count_df[trigger_state_col].unique()
    trigger_states = [TriggerState(value=ts) for ts in trigger_state_values]

    v0 = variant_values[0]
    if isinstance(v0, tuple):  # multi-experiment
        launches = [Launch(value=v) for v in variant_values if CATEG_NAN_VALUE not in v]
        non_control_launches = [Launch(value=v) for v in variant_values if CATEG_NAN_VALUE not in v and v != tuple([CONTROL_LABEL] * len(v))]
    else:  # simple experiment
        launches = [Launch(value=v) for v in variant_values if v != CATEG_NAN_VALUE]
        non_control_launches = [Launch(value=v) for v in variant_values if v != CATEG_NAN_VALUE and v != CONTROL_LABEL]

    # Add the trigger state count for each variant.
    # This will first determines which trigger state each variant belong too
    # and returns the total count
    # This is done by aggregation w.r.t `variant_col` and `transform` method.
    variant_count_df[TRIGGER_STATE_COUNT_COL] = variant_count_df[VARIANT_COUNT_COL].groupby(variant_count_df[trigger_state_col]).transform("sum")

    # Sets index for the dataframes to make it easier and faster to extract data.
    # For example in order to get the count of a variant,
    # we can simply execute `variant_count_df.at[variant_value, VARIANT_COUNT_COL]`
    variant_count_df.set_index(variant_col, inplace=True)
    trigger_state_count_df.set_index(trigger_state_col, inplace=True)

    total_count = variant_count_df[VARIANT_COUNT_COL].sum()
    total_triggered_count = variant_count_df.loc[variant_count_df[TRIGGER_STATE_OVERALL_COL], VARIANT_COUNT_COL].sum()
    total_triggered_percent = 100.0 * total_triggered_count / total_count

    variant_count_df[VARIANT_PERCENT_COL] = 100.0 * variant_count_df[VARIANT_COUNT_COL] / total_count
    variant_count_df[TRIGGER_STATE_PERCENT_COL] = 100.0 * variant_count_df[TRIGGER_STATE_COUNT_COL] / total_count

    # For each trigger state, we find the distribution of the variants with that state.
    # For example see the following two examples:
    # - the trigger state is (True, False).
    #       Then the corresponding variants could be these three:
    #       ("v1", "nan") and ("v2", "nan") and ("control", "nan")
    #       each covering 33.33% of the triggered (sums up to 100.00%).
    # - the trigger state is (True, True).
    #       Then correponding variants could be these six:
    #       ("v1", "enabled"), ("v2", "enabled"), ("control", "enabled"),
    #       ("v1", "control"), ("v2", "control"), ("control", "control")
    #       each covering variuos percentages which also sums up to 100.00%
    variant_count_df[VARIANT_OVER_TRIGGERED_PERCENT_COL] = 100.0 * variant_count_df[VARIANT_COUNT_COL] / variant_count_df[TRIGGER_STATE_COUNT_COL]

    trigger_state_count_df[TRIGGER_STATE_PERCENT_COL] = 100.0 * trigger_state_count_df[TRIGGER_STATE_COUNT_COL] / total_count

    conditional_trigger_dfs = None
    overlap_rates = None
    if isinstance(v0, tuple):
        num_expts = len(v0)
        conditional_trigger_dfs = {}
        overlap_rates = {}
        for i in range(len(v0)):
            ind = trigger_state_count_df.index.map(lambda x: x[i])
            df_slice = trigger_state_count_df.loc[ind]
            if len(df_slice) > 0:
                conditional_triggered_count = sum(df_slice[TRIGGER_STATE_COUNT_COL])
                df_slice.loc[:, TRIGGER_STATE_PERCENT_COL] = 100 * (df_slice[TRIGGER_STATE_COUNT_COL] / conditional_triggered_count)
                conditional_trigger_dfs[i + 1] = df_slice
                # Compute the amount of overlap for experiment i
                # The only possibility where there is no overlap is:
                # the case where other index than `i` is False
                x0 = [False] * num_expts
                x0[i] = True
                t0 = tuple(x0)
                ind = df_slice.index.map(lambda x: x != t0)
                overlap_rate = sum(df_slice.loc[ind][TRIGGER_STATE_PERCENT_COL])
                overlap_rates[i + 1] = overlap_rate

    return DerivedExptStats(
        variants=variants,
        trigger_states=trigger_states,
        launches=launches,
        non_control_launches=non_control_launches,
        variant_count_df=variant_count_df,
        trigger_state_count_df=trigger_state_count_df,
        total_count=total_count,
        total_triggered_count=total_triggered_count,
        total_triggered_percent=total_triggered_percent,
        conditional_trigger_dfs=conditional_trigger_dfs,
        overlap_rates=overlap_rates,
    )


def get_expt_stats(
    dc: DataContainer,
    variant_col: str = VARIANT_COL,
    trigger_state_col: str = TRIGGER_STATE_COL,
    io_param: Optional[IOParam] = None,
) -> DerivedExptStats:
    """
    This function computes the derived statistics for a multi-experiment (could be univariate as well).
    Note that this function assumes that each unit of the experiment appears on only one row of the input dataframe.
    It is generally OK, if this assumption is violated in a small percentage of the data.
    However, we delibrately do not pass an expt_untit_col to this function for simplicity and
    deduplication is done by the caller if needed.

    Args:
        dc: The DataContainer containing the experiment data.
            The data includes `variant_col` column.
        variant_col: The variant column (tuples)
        trigger_state_col: The trigger state column which will be produced.
            This will be a bool tuple denoting the trigger state.
        io_param: Optional IOParam instance, used to extract the Cursor instance
            if the data is in a SQL table/query.

    Returns:
        DerivedExptStats: The derived statistics for the experiment.

    Raises:
        ValueError: If a SQL source is provided without a Cursor inside io_param.
    """

    variant_count_df = calc_variant_count_df(
        dc=dc,
        variant_col=variant_col,
        io_param=io_param,
    )

    return get_expt_stats_vcd(
        variant_count_df=variant_count_df,
        variant_col=variant_col,
        trigger_state_col=trigger_state_col,
    )
