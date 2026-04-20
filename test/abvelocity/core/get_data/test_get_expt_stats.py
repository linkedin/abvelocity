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
# #ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# Original author: Reza Hosseini

import pandas as pd
import pytest
from abvelocity.core.get_data.data_container import DataContainer
from abvelocity.core.get_data.duckdb_cursor import DuckDBCursor
from abvelocity.core.get_data.get_expt_stats import get_expt_stats
from abvelocity.core.param.constants import CATEG_NAN_VALUE, CONTROL_LABEL
from abvelocity.core.param.io_param import IOParam
from abvelocity.core.param.launch import Launch
from abvelocity.core.param.variant import TriggerState, Variant


@pytest.fixture(scope="module")
def df():
    return pd.DataFrame(
        {
            "id": range(10),
            "variant": [
                ("a", "b", "c"),
                ("a", "b", "c"),
                ("e", "f", "g"),
                (CONTROL_LABEL, CONTROL_LABEL, CONTROL_LABEL),
                (CONTROL_LABEL, CONTROL_LABEL, CONTROL_LABEL),
                (CONTROL_LABEL, CONTROL_LABEL, CONTROL_LABEL),
                ("a", CATEG_NAN_VALUE, "d"),
                ("a", CATEG_NAN_VALUE, "d"),
                ("a", CATEG_NAN_VALUE, "g"),
                (CATEG_NAN_VALUE, CATEG_NAN_VALUE, CATEG_NAN_VALUE),
            ],
        }
    )


def check_correctness(res):
    """Correctness check is the same for the two following tests."""
    # Expected values
    expected_variants = [
        Variant(value=("a", "b", "c"), name="(a, b, c)"),
        Variant(value=("a", "nan", "d"), name="(a, nan, d)"),
        Variant(value=("a", "nan", "g"), name="(a, nan, g)"),
        Variant(value=("control", "control", "control"), name="(control, control, control)"),
        Variant(value=("e", "f", "g"), name="(e, f, g)"),
        Variant(value=("nan", "nan", "nan"), name="(nan, nan, nan)"),
    ]
    expected_launches = [
        Launch(value=("a", "b", "c"), name="(a, b, c)"),
        Launch(value=("control", "control", "control"), name="(control, control, control)"),
        Launch(value=("e", "f", "g"), name="(e, f, g)"),
    ]
    expected_non_control_launches = [
        Launch(value=("a", "b", "c"), name="(a, b, c)"),
        Launch(value=("e", "f", "g"), name="(e, f, g)"),
    ]
    expected_trigger_states = [
        TriggerState(value=(True, True, True), overall_value=True, name="(True, True, True)"),
        TriggerState(value=(True, False, True), overall_value=True, name="(True, False, True)"),
        TriggerState(value=(False, False, False), overall_value=False, name="(False, False, False)"),
    ]
    expected_variant_count_df = {
        "variant_count": [2, 2, 1, 3, 1, 1],
        "trigger_state": [
            (True, True, True),
            (True, False, True),
            (True, False, True),
            (True, True, True),
            (True, True, True),
            (False, False, False),
        ],
        "trigger_state_overall": [True, True, True, True, True, False],
        "trigger_state_count": [6, 3, 3, 6, 6, 1],
        "variant_percent": [20.0, 20.0, 10.0, 30.0, 10.0, 10.0],
        "trigger_state_percent": [60.0, 30.0, 30.0, 60.0, 60.0, 10.0],
        "variant_over_triggered_pcnt": [
            100.0 * 2.0 / 6.0,
            100.0 * 2.0 / 3.0,
            100.0 * 1.0 / 3.0,
            100.0 * 3.0 / 6.0,
            100.0 * 1.0 / 6.0,
            100.0 * 1.0 / 1.0,
        ],
    }
    expected_trigger_state_count_df = {
        "trigger_state_count": [1, 3, 6],
        "trigger_state_percent": [10.0, 30.0, 60.0],
    }
    expected_total_count = 10
    expected_total_triggered_count = 9
    expected_total_triggered_percent = 90.0

    # Create the expected DataFrame with the same index name as the actual DataFrame
    expected_variant_count_df = pd.DataFrame(
        expected_variant_count_df,
        index=[
            ("a", "b", "c"),
            ("a", "nan", "d"),
            ("a", "nan", "g"),
            ("control", "control", "control"),
            ("e", "f", "g"),
            ("nan", "nan", "nan"),
        ],
    )
    expected_variant_count_df.index.name = "variant"

    expected_trigger_state_count_df = pd.DataFrame(
        expected_trigger_state_count_df,
        index=[(False, False, False), (True, False, True), (True, True, True)],
    )
    expected_trigger_state_count_df.index.name = "trigger_state"

    # This addresses the non-deterministic order from the database query.
    res.variant_count_df.sort_index(inplace=True)
    expected_variant_count_df.sort_index(inplace=True)

    # You also need to sort the trigger_state_count_df for consistency
    res.trigger_state_count_df.sort_index(inplace=True)
    expected_trigger_state_count_df.sort_index(inplace=True)

    # Asserts
    assert sorted(res.variants) == sorted(expected_variants)
    assert sorted(res.launches) == sorted(expected_launches)
    assert sorted(res.non_control_launches) == sorted(expected_non_control_launches)
    assert sorted(res.trigger_states) == sorted(expected_trigger_states)
    pd.testing.assert_frame_equal(res.variant_count_df, expected_variant_count_df)
    pd.testing.assert_frame_equal(res.trigger_state_count_df, expected_trigger_state_count_df)
    assert res.total_count == expected_total_count
    assert res.total_triggered_count == expected_total_triggered_count
    assert res.total_triggered_percent == expected_total_triggered_percent


def test_get_expt_stats(df):
    res = get_expt_stats(DataContainer(pandas_df=df))
    check_correctness(res)


def test_get_expt_stats_sql_duckdb(df):
    """Same as first test but using DB."""

    # DuckDB setup
    cursor = DuckDBCursor()

    # Create the temporary table
    table_name = "test_expt_data"

    df = df.copy()
    df["variant"] = df["variant"].apply(lambda x: ", ".join(map(str, x)))

    # Register the raw DataFrame as a temporary view
    cursor._db_connection.register("test_expt_data_temp", df)

    create_table_query = f"""
    CREATE TABLE {table_name} AS
    SELECT
        id,
        STRING_SPLIT_REGEX(variant, ', ') AS variant
    FROM test_expt_data_temp;
    """

    cursor._db_connection.execute(create_table_query)
    cursor._db_connection.execute("DROP VIEW test_expt_data_temp;")  # Clean up the registered Pandas view

    # Setup DataContainer and IOParam
    io_param = IOParam(cursor=cursor)
    dc = DataContainer(table_name=table_name)

    # Call the function using the SQL path
    res = get_expt_stats(dc=dc, io_param=io_param)

    # Correctness checks
    check_correctness(res)

    # Clean up
    cursor.execute(f"DROP TABLE {table_name}")
    cursor._db_connection.close()


def test_get_expt_stats_conditional_trigger():
    df = pd.DataFrame(
        {
            "id": range(14),
            "variant": [
                ("a", "b", "c"),
                ("a", "b", "c"),
                ("e", "f", "g"),
                (CONTROL_LABEL, CONTROL_LABEL, CONTROL_LABEL),
                (CONTROL_LABEL, CONTROL_LABEL, CONTROL_LABEL),
                (CONTROL_LABEL, CONTROL_LABEL, CONTROL_LABEL),
                ("a", CATEG_NAN_VALUE, "d"),
                ("a", CATEG_NAN_VALUE, "d"),
                ("a", CATEG_NAN_VALUE, "g"),
                (CATEG_NAN_VALUE, CATEG_NAN_VALUE, "g"),
                (CATEG_NAN_VALUE, CATEG_NAN_VALUE, "g"),
                (CATEG_NAN_VALUE, CATEG_NAN_VALUE, "g"),
                (CATEG_NAN_VALUE, CATEG_NAN_VALUE, "g"),
                (CATEG_NAN_VALUE, CATEG_NAN_VALUE, CATEG_NAN_VALUE),
            ],
        }
    )

    res = get_expt_stats(DataContainer(pandas_df=df))

    expected_overlap_rates = {1: 100.0, 2: 100.0, 3: 69.2308}

    for i, rate in expected_overlap_rates.items():
        assert round(res.overlap_rates[i], 3) == round(rate, 3)
