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

from abvelocity.core.param.metric import Metric, UMetric, get_u_metrics


def test_umetric_creation1():
    umetric = UMetric(
        col="test_col",
        agg="SUM",
        fill_na=0,
        condition="condition1 AND condition2",
        name="test_metric",
    )
    assert umetric.col == "test_col"
    assert umetric.agg == "SUM"
    assert umetric.fill_na == 0
    assert umetric.condition == "condition1 AND condition2"
    assert umetric.name == "test_metric"


def test_metric_creation2():
    numerator = UMetric(
        col="numerator_col",
        agg="SUM",
        fill_na=0,
        condition="condition1",
        name="numerator_metric",
    )
    denominator = UMetric(
        col="denominator_col",
        agg="COUNT",
        fill_na=1,
        condition="condition2",
        name="denominator_metric",
    )
    metric = Metric(
        numerator=numerator,
        numerator_agg="SUM",
        denominator=denominator,
        denominator_agg="COUNT",
        name="test_metric",
    )
    assert metric.numerator == numerator
    assert metric.numerator_agg == "SUM"
    assert metric.denominator == denominator
    assert metric.denominator_agg == "COUNT"
    assert metric.name == "test_metric"


def test_metric_creation_sample_count():
    numerator = UMetric(
        col="numerator_col",
        agg="SUM",
        fill_na=0,
        condition="condition1",
        name="numerator_metric",
    )
    denominator = UMetric(
        col="denominator_col",
        agg="COUNT",
        fill_na=1,
        condition="condition2",
        name="denominator_metric",
    )
    sample_count = UMetric(col="eligible", agg="MAX", fill_na=0, name="sample_count")

    metric = Metric(
        numerator=numerator,
        numerator_agg="SUM",
        denominator=denominator,
        denominator_agg="COUNT",
        sample_count=sample_count,
        name="test_metric",
    )
    assert metric.numerator == numerator
    assert metric.numerator_agg == "SUM"
    assert metric.denominator == denominator
    assert metric.denominator_agg == "COUNT"
    assert metric.name == "test_metric"
    assert metric.sample_count == sample_count


def test_get_u_metrics():
    numerator1 = UMetric(
        col="numerator1_col",
        agg="SUM",
        fill_na=0,
        condition="condition1",
        name="numerator1_metric",
    )
    numerator2 = UMetric(
        col="numerator2_col",
        agg="COUNT",
        fill_na=1,
        condition="condition2",
        name="numerator2_metric",
    )
    denominator = UMetric(
        col="denominator_col",
        agg="COUNT",
        fill_na=1,
        condition="condition3",
        name="denominator_metric",
    )
    metrics = [
        Metric(
            numerator=numerator1,
            numerator_agg="SUM",
            denominator=denominator,
            denominator_agg="COUNT",
            name="metric1",
        ),
        Metric(numerator=numerator2, numerator_agg="COUNT", denominator=None, name="metric2"),
    ]
    umetrics = get_u_metrics(metrics)
    assert len(umetrics) == 3
    assert numerator1 in umetrics
    assert numerator2 in umetrics
    assert denominator in umetrics


def test_get_u_metrics_dedupe():
    numerator1 = UMetric(
        col="numerator1_col",
        agg="SUM",
        fill_na=0,
        condition="condition1",
        name="numerator1_metric",
    )
    numerator2 = UMetric(
        col="numerator2_col",
        agg="COUNT",
        fill_na=1,
        condition="condition2",
        name="numerator2_metric",
    )
    denominator = UMetric(
        col="denominator_col",
        agg="COUNT",
        fill_na=1,
        condition="condition3",
        name="denominator_metric",
    )
    metrics = [
        Metric(
            numerator=numerator1,
            numerator_agg="SUM",
            denominator=denominator,
            denominator_agg="COUNT",
            name="metric1",
        ),
        Metric(numerator=numerator2, numerator_agg="COUNT", denominator=None, name="metric2"),
        Metric(
            numerator=numerator1,  # repeated
            numerator_agg="COUNT",
            denominator=None,
            name="metric3",
        ),
    ]
    umetrics = get_u_metrics(metrics)
    assert len(umetrics) == 3
    assert numerator1 in umetrics
    assert numerator2 in umetrics
    assert denominator in umetrics
