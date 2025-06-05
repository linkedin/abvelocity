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

from dataclasses import dataclass
from typing import Optional

SUM = "SUM"
COUNT = "COUNT"
MAX = "MAX"
MIN = "MIN"
AVG = "AVG"
STDDEV = "STDDEV"
FIRST = "FIRST"
LAST = "LAST"
"""
These are the typical aggregation functions.
Any aggregation function which can be utilized with the SQL engine can work.
"""


@dataclass
class UMetric:
    """
    This dataclass specified how to calculate a unit-level metric (e.g. user level)
    which depends on one column of data only (`col`) given a table which might have
    multiple data for the given unit e.g. an event table, or a table wth data across
    various dates.

    The information here will prescribe how to aggregate the raw column data per unit via `agg`

    It also prescribes how to treat missing data for units which do not appear in the metric tables.
    An important use case is when the metric table if joined with expt assignment table.
    In such cases, lack of existence in metric table could mean no signup for example and
    the aggregate value should be zero, rather than None.

    We allow passing conditions as well which can be added to the SQL query when getting the
    metric data.
    """

    col: str
    """
    The column name in the metric table which contains the metric data.
    """
    agg: str = SUM
    """
    The aggregation function to be used to aggregate the column data within each unit.
    """
    fill_na: Optional[float] = None
    """
    The value to be used to fill missing data for units which do not appear in the metric table.
    """
    conditions: Optional[list[str]] = None
    """
    The conditions to be added to the SQL query when getting the metric data.
    """
    name: Optional[str] = None
    """
    The name of the metric.
    """

    def __post_init__(self):
        if self.name is None:
            self.name = self.col


@dataclass
class Metric:
    """
    This dataclass specifies how to calculate a metric which can be either a simple metric
    or a ratio of two metrics.

    The input metrics are univariate metrics (depend on one column only)
    and already been aggregated at unit level (within unit aggregation).

    We allow to apply flexible aggregation across units for the numerator and denominator metrics.

    Here are some example use cases:
        - Get signup rate: numerator is the number of signups, denominator is the number of users.

            ``raw_signup = UMetric(
                col="new_signup",
                agg="MAX",
                fill_na=0)

            signup_rate = Metric(
                numerator = raw_signup,
                denominator = raw_signup,
                numerator_agg = SUM,
                denominator_agg = COUNT)``

        - Get retention rate: numerator is the number of retained users, denominator is the number of users eligible for renew (target).

            ``retention_rate = Metric(
                numerator=UMetric(
                    col="n_renew",
                    agg=MAX,
                    fill_na=0),
                denominator=UMetric(
                    col="n_target",
                    agg=MAX,
                    fill_na=0),
                numerator_agg=SUM,
                denominator_agg=SUM)``
    """

    numerator: UMetric
    """
    The numerator metric.
    """
    numerator_agg: str = SUM
    """
    The aggregation function to be used to aggregate the numerator metric across units.
    """
    denominator: Optional[UMetric] = None
    """
    The denominator metric.
    """
    denominator_agg: str = SUM
    """
    The aggregation function to be used to aggregate the denominator metric across units.
    """
    sample_count: Optional[UMetric] = None
    """
    This specifies a way to count units (sample size) if the number of rows is not correct.
    Here is an example where this can be useful.
    Assume we like to estimate the impact of an experiment on retention.
    The unit data might be as follows:

    unit, renew, eligible
    ---------------------
    u1,   1,     1
    u2,   1,     1
    u3,   0,     1
    u4,   0,     0

    Note that in this case, u4 is not even eligible for renew, but if we count rows
    and use a binomial based or appromixation of it, it will be counted in the sample size.
    In this case the user can pass eligible as a `UMetric` via this field, so that we get zeros and
    ones for this column.
    """
    name: Optional[str] = None
    """
    The name of the metric.
    """

    def __post_init__(self):
        if self.name is None:
            if self.denominator is None:
                self.name = self.numerator.name
            else:
                self.name = f"{self.numerator.name}/{self.denominator.name}"


def get_u_metrics(metrics=list[Metric]) -> list[UMetric]:
    """
    This function gets the list of UMetrics from the list of Metrics.
    The purpose is to gather all `UMetric`s needed for `Metric` calculations.
    It also dedupes the metrics in the process.

    Args:
        metrics: A list of `Metric`s, each of which can include `numerator`,
            `denominator` and `sample_count` etc.

    Returns:
        result: A list of `UMetric`s.

    """
    umetrics = []
    # We store observed `UMetric`s based on their `col`, `agg`, `name` in tuples.
    # Then we only add them if they are not addeded yet.
    added_u_metrics = set()

    def add_u_metric(u_metric: UMetric) -> None:
        """
        Adds a UMetric only if it's not added according to its tuple repr
        in terms of the three fields of col, agg anf name.
        Args:
            u_metric: A UMetric

        Alters:
            umetrics
        """
        u_metric_tuple_repr = (u_metric.col, u_metric.agg, u_metric.name)
        if u_metric_tuple_repr not in added_u_metrics:
            umetrics.append(u_metric)
            added_u_metrics.add(u_metric_tuple_repr)

    for metric in metrics:
        add_u_metric(metric.numerator)
        if metric.denominator is not None:
            add_u_metric(metric.denominator)
        if metric.sample_count is not None:
            add_u_metric(metric.sample_count)
    return umetrics
