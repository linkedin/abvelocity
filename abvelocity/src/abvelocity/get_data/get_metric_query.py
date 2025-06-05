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

from typing import Optional

from abvelocity.get_data.query_info import QueryInfo
from abvelocity.param.constants import UNIT_COL
from abvelocity.param.metric import UMetric


class GetMetricQuery:
    """
    This class is used to get metric data from the database.
    The class is initialized with the table name and then the query is constructed via the `construct_query` method.
    User can access the query for usage or use the `get_data` method to get the data.

    Attributes:
        table_name: The name of the table to get the metric data from.
        query: The query to get the metric data.
    """

    def __init__(self, table_name: str, date_col: str, metric_table_unit_col: Optional[str] = None):
        """
        Args:
            table_name: The name of the table to get the metric data from.
        """
        self.table_name = table_name
        self.date_col = date_col
        self.metric_table_unit_col = metric_table_unit_col
        self.query = None

        if not self.metric_table_unit_col:
            self.metric_table_unit_col = UNIT_COL

    def construct_query(
        self,
        start_date: str,
        end_date: str,
        u_metrics: Optional[list[UMetric]] = None,
        condition: Optional[str] = None,
    ) -> str:
        """
        This method constructs the query to get the metric data.
        Args:
            start_date: The start of the overlap period between the expts.
            end_date: The end of the overlap period between the expts.
            u_metrics: The list of unit-level metrics of interest.
                If None, we use "*" to extract all fields.
                If not None, we query the specified u_metrics plus `UNIT_COL`.
                For each u_metric e.g.:
                    `UMetric(agg="SUM", col="n_RandomProduct_sessions", name="n_RandomProduct_sessions")`
                A string like this is constructed: `SUM(n_RandomProduct_sessions) AS n_RandomProduct_sessions`
            condition: `str` or None, default None.
                A SQL "WHERE" clause to be added to the query.
                One use case is to subsample the users.

        Returns:
            sql_query_result: SQL query result which includes signup data in its "df" attribute.
                See `~abvelocity.get_data.get_sql_df.SqlQueryResult` for details.
        """
        if u_metrics is None:
            columns = "*"
            aggregations = None
        else:
            columns = [f"{self.metric_table_unit_col} AS {UNIT_COL}"]
            aggregations = [f"{u.agg}({u.col}) AS {u.name}" for u in u_metrics]

        # Below in the query we get member_id as memberid to become consistent with expt
        time_condition = f"""{self.date_col} BETWEEN '{start_date}' AND '{end_date}'"""
        conditions = [time_condition]

        if condition is not None:
            # We need to do a replacement since unit column is different
            # in signups data.
            condition = condition.replace(UNIT_COL, self.metric_table_unit_col)
            conditions += f"""{condition}"""

        query_info = QueryInfo(
            table_name=self.table_name,
            columns=columns,
            aggregations=aggregations,
            conditions=conditions,
        )

        self.query = query_info.construct()

        return self.query
