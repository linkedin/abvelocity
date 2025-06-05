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

from dataclasses import dataclass, field
from typing import Callable, Optional

from abvelocity.param.metric import Metric


@dataclass
class MetricFamily:
    """This is a dataclass to store information on supported families of metrics.
    Such information is then used to get data.
    In order to add a new metric family developers need to:
        - (1) implement the get and process functions.
        - (2) add metric name to `~abvelocity.get_data.li_metric_families.METRIC_FAMILY_MAP`.
    """

    name: str
    """Metric family name."""
    get_metric_query: Optional[Callable] = None
    """Function to get a metric (SQL) query."""
    metrics: Optional[list[Metric]] = None
    """Default Metrics for the table."""
    get_metric_query_params: dict = field(default_factory=lambda: {})
    """Optional parameters to pass to `get_metric_query`."""
    process_expt_metric_df: Optional[Callable] = None
    """Optional function to post process the joined experiment and metric data."""
    process_expt_metric_df_params: dict = field(default_factory=lambda: {})
    """Optional parameters to pass to `process_expt_metric_df`."""
