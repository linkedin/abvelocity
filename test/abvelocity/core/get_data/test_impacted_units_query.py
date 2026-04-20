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
#
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
from abvelocity.core.get_data.duckdb_cursor import DuckDBCursor
from abvelocity.core.get_data.impacted_units_query import ImpactedUnitsQuery
from abvelocity.core.param.expt_info import ExptInfo, MultiExptInfo
from abvelocity.core.testing.assert_query_is_equal import assert_query_is_equal

EXPT1_QUERY = "SELECT member_id, variant FROM assignments WHERE test_key = 'expt1'"
EXPT2_QUERY = "SELECT member_id, variant FROM assignments WHERE test_key = 'expt2'"


def test_single_experiment():
    multi_expt_info = MultiExptInfo(expt_info_list=[ExptInfo(expt_unit_col="member_id", query=EXPT1_QUERY)])
    result = ImpactedUnitsQuery(multi_expt_info).construct()

    expected = f"""
        SELECT DISTINCT member_id
        FROM (
            SELECT member_id
            FROM (
                {EXPT1_QUERY}
            ) AS expt_0
            WHERE member_id IS NOT NULL
        ) AS impacted_units
    """
    assert_query_is_equal(result, expected)


def test_two_experiments_use_union_all():
    multi_expt_info = MultiExptInfo(
        expt_info_list=[
            ExptInfo(expt_unit_col="member_id", query=EXPT1_QUERY),
            ExptInfo(expt_unit_col="member_id", query=EXPT2_QUERY),
        ]
    )
    result = ImpactedUnitsQuery(multi_expt_info).construct()

    expected = f"""
        SELECT DISTINCT member_id
        FROM (
            SELECT member_id FROM (
                {EXPT1_QUERY}
            ) AS expt_0
            WHERE member_id IS NOT NULL
            UNION ALL
            SELECT member_id FROM (
                {EXPT2_QUERY}
            ) AS expt_1
            WHERE member_id IS NOT NULL
        ) AS impacted_units
    """
    assert_query_is_equal(result, expected)


def test_unit_col_inherited_from_multi_expt_info():
    multi_expt_info = MultiExptInfo(
        expt_info_list=[ExptInfo(query=EXPT1_QUERY), ExptInfo(query=EXPT2_QUERY)],
        expt_unit_col="member_id",
    )
    result = ImpactedUnitsQuery(multi_expt_info).construct()

    expected = f"""
        SELECT DISTINCT member_id
        FROM (
            SELECT member_id FROM (
                {EXPT1_QUERY}
            ) AS expt_0
            WHERE member_id IS NOT NULL
            UNION ALL
            SELECT member_id FROM (
                {EXPT2_QUERY}
            ) AS expt_1
            WHERE member_id IS NOT NULL
        ) AS impacted_units
    """
    assert_query_is_equal(result, expected)


def test_construct_stores_query():
    multi_expt_info = MultiExptInfo(expt_info_list=[ExptInfo(expt_unit_col="member_id", query=EXPT1_QUERY)])
    obj = ImpactedUnitsQuery(multi_expt_info)
    assert obj.query is None
    obj.construct()
    assert obj.query is not None


def test_empty_expt_info_list_raises():
    with pytest.raises(ValueError, match="must not be empty"):
        ImpactedUnitsQuery(MultiExptInfo(expt_info_list=[])).construct()


def test_missing_expt_unit_col_raises():
    with pytest.raises(ValueError):
        ImpactedUnitsQuery(MultiExptInfo(expt_info_list=[ExptInfo(query=EXPT1_QUERY)])).construct()


@pytest.fixture
def duckdb_cursor():
    cursor = DuckDBCursor(max_retries=1)
    expt1_df = pd.DataFrame(
        {
            "member_id": [1, 2, 3, 4],
            "variant": ["enabled", "enabled", "control", "control"],
        }
    )
    expt2_df = pd.DataFrame(
        {
            "member_id": [3, 4, 5, 6],
            "variant": ["enabled", "enabled", "control", "control"],
        }
    )
    for df, table in [(expt1_df, "expt1_asgmt"), (expt2_df, "expt2_asgmt")]:
        cursor._db_connection.register(f"{table}_view", df)
        cursor._db_connection.execute(f"CREATE TABLE {table} AS SELECT * FROM {table}_view")
        cursor._db_connection.unregister(f"{table}_view")
    yield cursor
    cursor.close()


def test_get_pandas_df_returns_distinct_impacted_unit_ids(duckdb_cursor):
    """Two overlapping experiments: union = {1,2,3,4} ∪ {3,4,5,6} = {1,2,3,4,5,6} (6 distinct)."""
    multi_expt_info = MultiExptInfo(
        expt_info_list=[
            ExptInfo(expt_unit_col="member_id", query="SELECT member_id, variant FROM expt1_asgmt"),
            ExptInfo(expt_unit_col="member_id", query="SELECT member_id, variant FROM expt2_asgmt"),
        ]
    )

    df = ImpactedUnitsQuery(multi_expt_info).get_pandas_df(duckdb_cursor)

    expected = pd.DataFrame({"member_id": [1, 2, 3, 4, 5, 6]})
    pd.testing.assert_frame_equal(
        df.sort_values("member_id").reset_index(drop=True),
        expected,
    )
