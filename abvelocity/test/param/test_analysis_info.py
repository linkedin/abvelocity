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

from abvelocity.param.analysis_info import AnalysisInfo
from abvelocity.param.expt_info import ExptInfo, MultiExptInfo
from abvelocity.param.metric import Metric, UMetric
from abvelocity.param.metric_info import MetricInfo


def test_analysis_info_initialization():
    expt_info1 = ExptInfo(test_key="test123", experiment_id=101)
    expt_info2 = ExptInfo(test_key="test456", experiment_id=102)
    multi_expt_info = MultiExptInfo(expt_info_list=[expt_info1, expt_info2])

    metric1 = Metric(name="metric1", numerator=UMetric(col="clicks"))
    metric2 = Metric(name="metric2", numerator=UMetric(col="payments"))

    analysis_info = AnalysisInfo(
        multi_expt_info=multi_expt_info, metric_info_list=[MetricInfo(metrics=[metric1, metric2])]
    )

    assert len(analysis_info.multi_expt_info.expt_info_list) == 2
    assert len(analysis_info.metric_info_list[0].metrics) == 2
    assert analysis_info.metric_info_list[0].metrics[0].name == "metric1"
    assert analysis_info.metric_info_list[0].metrics[1].name == "metric2"


def test_analysis_info_no_metrics():
    expt_info1 = ExptInfo(test_key="test123", experiment_id=101)
    expt_info2 = ExptInfo(test_key="test456", experiment_id=102)
    multi_expt_info = MultiExptInfo(expt_info_list=[expt_info1, expt_info2])

    analysis_info = AnalysisInfo(multi_expt_info=multi_expt_info)

    assert analysis_info.metric_info_list is None
