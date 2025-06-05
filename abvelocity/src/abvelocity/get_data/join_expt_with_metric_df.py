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

"""The goal is to merge various experiment assignment DataContainers."""
import gc
import resource
import sys
from typing import Optional

from abvelocity.get_data.data_container import DataContainer
from abvelocity.param.constants import TRIGGER_TIME_COL, UNIT_COL
from abvelocity.param.metric import UMetric
from abvelocity.utils.check_df_validity import check_df_validity

ON_COLS = [UNIT_COL]
"""We commonly join on this variable."""


def join_expt_with_metric_df(
    expt_dc: DataContainer,
    metric_dc: DataContainer,
    on_cols: list[str] = ON_COLS,
    u_metrics: Optional[list[UMetric]] = None,
    how: str = "left",
    drop_time_cols: bool = True,
) -> DataContainer:
    """This function joins the experiment assignment DataContainer with metrics DataContainer.
    This is simply only a left join with experiment assignment data being on
    the left. This is because we only want to analyze / measure impact on
    (all) units who are seen in the experiment.
    We fill in the NAN values for each `u_metric` with its corresponding `u_metric_fill_na`.

    Args:
        expt_dc : The DataContainer including the experiment assignment information.
            It is to include `UNIT_COL` which is the unit of experimentation.
        metric_dc : The DataContainer including the experiment assignment information.
            It is to include `UNIT_COL` which is the unit of experimentation.
        u_metrics : The list of unit-level metrics of interest.
            This is used only to fill in the NAN values for each `u_metric` with its corresponding `u_metric.fill_na`.
            If None, no filling will be performed.
        on_cols : The list of columns to join on. The default is a list with only `UNIT_COL`.
        how : Specifies the method for merging. It can be "outer", "right" or "left".
            Usually we expect this to be a left join (expt data are on left).
        drop_time_cols: If True, we drop the time columns to save memory before the join.

    Returns:
        A DataContainer containing the joined data which includes experiment unit level data.

    Raises:
        None.
    """

    expt_df = expt_dc.df
    metric_df = metric_dc.df

    if expt_df is None or metric_df is None:
        raise ValueError("Input DataContainers must have non-None DataFrames.")

    # Data validation: Checks if there are necessary columns missing.
    err_trigger_source = "join_expt_with_metric_df"
    # Validation for `expt_df`.
    check_df_validity(
        df=expt_df,
        needed_cols=on_cols,
        err_trigger_source=err_trigger_source,
        err_message=f"`expt_df` has missing columns for merge (`on_cols`): {on_cols}.",
    )
    # Validation for `metric_df`.
    check_df_validity(
        df=metric_df,
        needed_cols=on_cols,
        err_trigger_source=err_trigger_source,
        err_message="`metric_df` has missing columns for merge (`on_cols`): {on_cols}.",
    )

    if drop_time_cols:
        # First drops time columns from experiment data, if they exist.
        for col in expt_df.columns:
            if TRIGGER_TIME_COL in col:
                print(f"column {col} is dropped from expt_df")
                del expt_df[col]

    mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)  # Convert to GB
    print(
        f"***\n Memory used in join_expt_with_metric_df (before joining expt_df and metric_df): {mem_usage:.4f} GB"
    )

    # We do a left join betweeen the unit (e.g. member) data and the u_metric data.
    # This is because we want to keep:
    # all units who have been seen in at least one of the experiments.
    df = expt_df.merge(metric_df, how=how, on=ON_COLS)

    mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)  # Convert to GB
    print(
        f"***\n Memory used in join_expt_with_metric_df after joining expt_df and metric_df: {mem_usage:.4f} GB"
    )

    # Free memory
    del expt_df
    del metric_df

    # Trigger garbage collection to reclaim memory
    gc.collect()

    mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)  # Convert to GB
    print(f"***\n Memory used in join_expt_with_metric_df after gc: {mem_usage:.4f} GB")

    # We fill in the NAN values for each u_metric with its corresponding `u_metric_fill_na`.
    # This is because if the unit does not have any u_metric data, assumptions are to be made on the metric.
    # Note that we cannot exclude such users from calculations.
    # An an example: If a user does not appear in signup data, the signup metric should be zero rather than None.
    if u_metrics is not None:
        for u_metric in u_metrics:
            if u_metric.fill_na is not None:
                df[u_metric.name] = df[u_metric.name].fillna(u_metric.fill_na)

    size_in_megabytes = sys.getsizeof(df) / 10**6
    print(f"`joined expt & metric df` size in MB: {size_in_megabytes}")

    return DataContainer(df=df, is_df=True)
