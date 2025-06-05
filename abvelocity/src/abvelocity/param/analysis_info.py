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

from abvelocity.param.expt_info import MultiExptInfo
from abvelocity.param.metric_info import MetricInfo


@dataclass
class AnalysisInfo:
    """
    This is a dataclass to contain the information for the experiments (at least 2)
    in the analysis as well as the metrics in the analysis.
    """

    multi_expt_info: MultiExptInfo
    """
    The list of experiments to be analyzed together.
    """
    metric_info_list: Optional[list[MetricInfo]] = None
    """
    The information for metrics given in a list.
    Each element is a `MetricInfo` which will include a list of metrics itself.
    As an example one such `MetricInfo` can contain signup metrics and another can
    contain engagement metrics.
    """
    start_date: Optional[str] = None
    """This is the analysis start data."""
    end_date: Optional[str] = None
    """This is the analysis end data."""

    def __post_init__(self):
        """Infers `metrics` from `metric_family` if available."""
        # Checks if `metric_info.metrics` is None but `metric_info.metric_family.metrics` is avaliable.
        if self.metric_info_list is not None:
            for metric_info in self.metric_info_list:
                if (
                    not metric_info.metrics
                    and metric_info.metric_family
                    and metric_info.metric_family.metrics
                ):
                    metric_info.metrics = metric_info.metric_family.metrics

    def __str__(self):
        metrics_str = ""
        for metric_info in self.metric_info_list:
            if metric_info.metrics:
                metrics_str += (
                    "\n" + ", ".join(str(metric) for metric in metric_info.metrics) + "\n"
                )

        metric_family_str = ""
        for metric_info in self.metric_info_list:
            if metric_info.metric_family:
                metric_family_str += f"_{metric_info.metric_family.name}"

        return (
            f"Analysis Information:\n"
            f"{self.multi_expt_info}\n"
            # f"- Metrics: {metrics_str}\n"
            f"- Metric Family: {metric_family_str}"
            f"- Analysis start date: {self.start_date}"
            f"- Analysis end date: {self.end_date}"
        )

    def to_html(self):
        metrics_html = ""
        for metric_info in self.metric_info_list:
            if metric_info.metrics:
                metrics_html += (
                    "<p>" + ",<p> ".join(str(metric) for metric in metric_info.metrics) + "<p>"
                )

        metric_family_html = ""
        for metric_info in self.metric_info_list:
            if metric_info.metric_family:
                metric_family_html += f"_{metric_info.metric_family.name}"

        return (
            f"<div>"
            f"<h3>Analysis Information</h3>"
            f"{self.multi_expt_info.to_html()}"
            f"<p><strong>Metric Family:</strong> {metric_family_html}</p>"
            # f"<p><strong>Metrics:</strong> {metrics_html}</p>"
            f"<p><strong>Analysis Start Date:</strong> {self.start_date}</p>"
            f"<p><strong>Analysis End Date:</strong> {self.end_date}</p>"
            f"</div>"
        )

    def to_markdown(self) -> str:
        """
        Convert the AnalysisInfo data to a Markdown format.
        """
        metrics_md = ""
        for metric_info in self.metric_info_list:
            if metric_info.metrics:
                metrics_md += "\n" + ", ".join(str(metric) for metric in metric_info.metrics) + "\n"

        metric_family_md = ""
        for metric_info in self.metric_info_list:
            if metric_info.metric_family:
                metric_family_md += f"_{metric_info.metric_family.name}"

        return (
            f"### Analysis Information\n"
            f"{self.multi_expt_info.to_markdown()}\n"
            f"**Metric Family**: {metric_family_md}\n"
            # f"**Metrics**: {metrics_md}\n"
            f"**Analysis Start Date**: {self.start_date}\n"
            f"**Analysis End Date**: {self.end_date}\n"
        )
