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
"""This module includes dataclasses to for parameters used in analysis of experiments."""

from dataclasses import dataclass
from typing import Optional

from abvelocity.core.param.metric import Metric
from abvelocity.core.param.metric_family import MetricFamily
from mashumaro.config import BaseConfig
from mashumaro.mixins.json import DataClassJSONMixin
from mashumaro.types import SerializationStrategy


class _SkipAsNone(SerializationStrategy):
    def serialize(self, value) -> None:
        return None

    def deserialize(self, value) -> None:
        return None


@dataclass
class MetricInfo(DataClassJSONMixin):
    """
    This is a dataclass to contain the information for a group of metrics.
    """

    metric_family: Optional[MetricFamily] = None
    """
    The metric family for the analysis.

        - This essentially says which table the metrics belong too and includes extra
        processing functions if needed (this feature should be rarely needed).

        - It also might include `metrics` field available which can be used
        to fill in `metrics` field in this dataclass.

    """
    metrics: Optional[list[Metric]] = None
    """
    metrics to be investigated.
    This might be inferred from `metric_family` if this field is not passed.
    """
    dims: Optional[list[str]] = None
    """
    Dimensions to include in the query.
    This might be inferred from `metric_family` if this field is not passed.
    """
    condition: Optional[str] = None
    """If passed the condition will be applied to all metrics."""
    start_date: Optional[str] = None
    """This is the analysis start data."""
    end_date: Optional[str] = None
    """This is the analysis end data."""

    class Config(BaseConfig):
        serialization_strategy = {MetricFamily: _SkipAsNone()}

    def __post_init__(self):
        """Infers `metrics` from `metric_family` if available."""
        # Checks if `self.metrics` is None but `self.metric_family.metrics` is avaliable.
        if not self.metrics and self.metric_family and self.metric_family.metrics:
            self.metrics = self.metric_family.metrics

        # If a condition is passed it will be applied to all metrics.
        # Individual condition should be passed to the metric itself.
        if self.condition:
            print(
                "\n***: In `MetricInfo.__post_init__`, condition: {self.condition}" f" is applied to metrics in metric_family.name: {self.metric_family.name}"
            )
            for metric in self.metrics:
                metric.condition = self.condition

    def __str__(self):
        metrics_str = "\n" + ", ".join(str(metric) for metric in self.metrics) + "\n" if self.metrics else "None"

        metric_family_str = None
        if self.metric_family:
            self.metric_family_str = self.metric_family.name

        return f"- Metrics: {metrics_str}\n" f"- Metric Family: {metric_family_str}"

    def to_html(self):
        """
        Convert the MetricInfo data to a html format.
        """
        metrics_html = "<p>" + ",<p> ".join(str(metric) for metric in self.metrics) + "<p>" if self.metrics else "None"

        metric_family_html = "None"
        if self.metric_family and self.metric_family.name:
            self.metric_family_str = self.metric_family.name

        return (
            f"<div>"
            f"<h3>Metric Information</h3>"
            f"<p><strong>Metric Family:</strong> {metric_family_html}</p>"
            f"<p><strong>Metrics:</strong> {metrics_html}</p>"
            f"</div>"
        )

    def to_markdown(self) -> str:
        """
        Convert the MetricInfo data to a Markdown format.
        """
        metrics_md = "\n" + ", ".join(str(metric) for metric in self.metrics) + "\n" if self.metrics else "None"

        metric_family_md = "None"
        if self.metric_family and self.metric_family.name:
            metric_family_md = self.metric_family.name

        return f"**Metric Family**: {metric_family_md}\n" f"**Metrics**: {metrics_md}\n"
