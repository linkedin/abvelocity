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

from abvelocity.core.param.metric import get_u_metrics
from abvelocity.core.param.metric_info import MetricInfo


def get_u_metrics_query_from_info(metric_info: MetricInfo, start_date: str, end_date: str, condition: Optional[str] = None) -> str:
    """
    This function constructs the query to get metric data from the database based on the metric info.
    It utilizes the `u_metrics_query` function of the `MetricFamily` class to construct
    the query.

    Args:
        metric_info: The metric info object that contains the information about the metric.
        start_date: The start date of the metric data to get.
        end_date: The end date of the metric data to get.
        condition: The condition to apply to the metric query. This is optional and can be used to apply additional filters to the metric data.

    Returns:
        The query to get the metric data from the database.
    """

    metric_family = metric_info.metric_family
    metric_join_unit_col = metric_family.metric_join_unit_col
    # Info such as table name for metrics is stored in
    # `u_metrics_query` which depends (only) on metric_family
    u_metrics_query = metric_family.u_metrics_query

    u_metrics_query_params = metric_family.u_metrics_query_params

    # Dimensions to include in the query (inferred from metric_family in __post_init__ if not passed)
    dims = metric_info.dims

    # Metric data
    u_metrics = get_u_metrics(metric_info.metrics)

    # If `condition` is passed through `metric_info`, it will be augmented
    # to current condition and then used by `u_metrics_query.construct`
    if metric_info.condition:
        print("\n***: In `get_mea_data` metric_info.condition: " f"{metric_info.condition} was applied to metrics in metric family: {metric_family.name}")

        # If condition already exists just augment it
        if condition:
            condition += f" AND {metric_info.condition}"
        else:
            # If condition is None, then `metric_info.condition`
            # becomes our condition
            condition = metric_info.condition

    if u_metrics is None:
        raise ValueError("metrics cannot be None.")
    print(f"\n*** u_metrics:\n{u_metrics}")

    if dims is not None:
        print(f"\n*** dims:\n{dims}")

    metric_query = u_metrics_query.construct(
        start_date=start_date,
        end_date=end_date,
        metric_join_unit_col=metric_join_unit_col,
        u_metrics=u_metrics,
        condition=condition,
        dims=dims,
        **u_metrics_query_params,
    )

    return metric_query
