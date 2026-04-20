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

from typing import List, Optional

# The UMetric class is already imported and available
from abvelocity.core.param.metric import UMetric


def join_expt_with_metric_queries(
    expt_assignment_query: str,
    metric_query: str,
    expt_unit_col: str,
    metric_join_unit_col: str,
    u_metrics: Optional[List[UMetric]] = None,
) -> str:
    """
    Generates a SQL query to perform a LEFT JOIN between two subqueries,
    handling fill_na logic for metrics.

    Args:
        expt_assignment_query: The SQL query for the experiment assignments.
        metric_query: The SQL query for the metrics.
        expt_unit_col: The unit column of the experiment assignment data.
        metric_join_unit_col: The unit column of the metric data.
        u_metrics: An optional list of UMetric objects to define which metrics
                   to select and how to handle NULL values.

    Returns:
        A single SQL query string that combines the two inputs with a LEFT JOIN.
    """
    # Build the list of columns to select from the metrics data
    if u_metrics:
        metric_select_list = []
        for metric in u_metrics:
            if metric.fill_na is not None:
                # Use COALESCE to replace NULL with the fill_na value
                # Handle string literals by wrapping in single quotes
                fill_val = f"'{metric.fill_na}'" if isinstance(metric.fill_na, str) else str(metric.fill_na)
                metric_select_list.append(f"COALESCE(metric_data.{metric.name}, {fill_val}) AS {metric.name}")
            else:
                metric_select_list.append(f"metric_data.{metric.name}")

        # Combine the selected metrics into a single string
        metric_select_str = ",\n        ".join(metric_select_list)
        select_clause = f"expt_data.*,\n        {metric_select_str}"
    else:
        # If no u_metrics are provided, select all columns from metric data
        select_clause = "expt_data.*,\n        metric_data.*"

    # Create the ON condition string from the list of columns
    on_condition = f"expt_data.{expt_unit_col} = metric_data.{metric_join_unit_col}"

    # Construct the final SQL query using CTEs for clarity and optimization
    query = f"""
WITH
    expt_data AS (
        {expt_assignment_query}
    ),
    metric_data AS (
        {metric_query}
    )
SELECT
    {select_clause}
FROM
    expt_data
LEFT JOIN
    metric_data
ON
    {on_condition}
    """
    return query
