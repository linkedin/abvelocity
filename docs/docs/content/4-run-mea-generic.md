# Run MEA Generic

**Author**: Reza Hosseini

You can use MEA directly on a joined dataframe which has assignment data and metric data in its columns.
This assumes the data is already joined. This implies that the data fits in the memory.

- Install the library using appropriate install command in the corresponding environment e.g.

```bash
pip install abvelocity
```

- Here we show the usage through simulated data.
- The resulting analysis report is here:
    - [HTML](/docs/static//test-results/mea/mea_sim1/mea_test.html)
    - [PDF](/docs/static//test-results/mea/mea_sim1/mea_test.pdf)

- Here are the steps for the simulation:
    - Simulate a heterogeneous population with varying attributes (`attribute_weights`): level, country, device
    - For each item in the population, we assign a baseline metrics (`metric_attribute_values`)
    - This will also act as a random baseline for each item
    - Two metrics, `metric1` and `metric2` are considered below
    - Then we simulate behavior from two experiments.
    - The assignment weights for various variants is defined by `expt_variant_weights_multi`
    - For Expt 1, we have variants: control, v1, v2
    - For Expt 2, we have variants: control, enabled
    - Then we simulate some univariate effects for each experiment by `expt_metric_impacts`
    - We deliberately only impose interaction effect on `metric1`
    - Based on univariate impacts for Expt 1:
        - for `metric1` the best variant is `v2 (+5)`
        - for `metric2` the best variant is `v2 (+5)`
    - Based on univariate impacts for Expt 2:
        - for `metric1` the best variant is `control` because enabled is -2
        - for `metric2` the best variant is `enabled` because enabled is +1
    - Note that due to `non_trigger_pct_multi=[5, 5]` argument both experiments almost trigger on all the population
    - Each trigger on random 95% of the population (close to 100%)
    - In the report [HTML](/docs/static//test-results/mea/mea_sim1/mea_test.html),
    we do see that the overlap rate for each experiment by the other is close to 95%.
    - But there is infact some areas where only one experiment overlaps.
    - If we ignore interactions and assuming both experiments trigger on all population (not exactly true), we would expect to see:
        - for metric1 the best combination is (v2, control) with delta being close to: +5
        - for metric2 the best combination is (v2, enabled), with delta being close to: +6
    - For metric2 due to no interactions we do expect to see a delta close to +5 for (v2, enabled)
    - This is confirmed by the report
    - However, the results should not hold for metric1 due to interaction and we should expect:
        - The best combination is (v1, enabled) with delta close to
            - delta(v1) + delta(enabled) + delta((v1, v2)) =

            ```python
            (-2) + (-2) + 15 = 11
            ```
            A more accurate expected value can be obtained by noticing:
            - The weight of population both Expts trigger: `0.95*0.95 = 0.9025`
            - The weight of population only Expt 1 triggers: `0.95 - 0.9025 = 0.0475`
            - The weight of population only Expt 2 triggers: `0.95 - 0.9025 = 0.0475`
            ```python
            expected_delta = ((-2)*(0.0475) + (-2)*(0.0475) + 11*0.9025) / (0.0475 + 0.0475 + 0.9025) = 9.76
            ```
            - therefore we expect to see a delta close to 11 (smaller because they do not trigger on all population)
        - In the below code block, we have implemented a helper function to calculate the expected delta for this simple
        case where there are only two experiments. The calculation in the function is similar to what we discussed above using
        the example.

```python
import os
from pathlib import Path

import numpy as np
import pandas as pd

from abvelocity.get_data.data_container import DataContainer
from abvelocity.get_data.join_expt_dfs import join_expt_dfs
from abvelocity.get_data.join_expt_with_metric_df import join_expt_with_metric_df
from abvelocity.mea.mea import MEA, MEAMetricResult, MEAResult
from abvelocity.param.analysis_info import AnalysisInfo
from abvelocity.param.constants import TRIGGER_TIME_COL, UNIT_COL, VARIANT_COL
from abvelocity.param.expt_info import ExptInfo, MultiExptInfo
from abvelocity.param.launch import Launch
from abvelocity.param.metric import Metric, UMetric
from abvelocity.param.metric_info import MetricInfo
from abvelocity.param.variant import ComparisonPair, Variant, VariantList
from abvelocity.sim.sim import Sim


attribute_weights = {
    "Level": {"Senior": 0.5, "Junior": 0.5},
    "Country": {"Iran": 0.1, "Canada": 0.3, "UK": 0.2, "US": 0.4},
    "Device": {"Smartphone": 0.6, "Tablet": 0.3, "Laptop": 0.1},
}
"""Example Define weights for attributes of the population."""


metric_attribute_values = {
    "metric1": {"Level": {"Junior": 5, "Senior": 2}, "Country": {"Iran": +5}},
    "metric2": {"Country": {"USA": 5, "Canada": 2}},
}
"""Example baseline metrics which depends on the attributes (but not experiment assignments)."""

expt_variant_weights_multi = [
    {"control": 0.4, "v1": 0.3, "v2": 0.3},  # Experiment 1
    {"control": 0.4, "enabled": 0.6},  # Experiment 2
]

# Univariate Effects
# For metric1: v1 is bad; enabled is bad. v2 is good.
# For metric2: v2 is good. control is good.
expt_metric_impacts = [
    {
        "control": {"metric1": 0, "metric2": 0},
        "v1": {"metric1": -2, "metric2": 0},
        "v2": {"metric1": +5, "metric2": +5},
    },  # Experiment 1 impact
    {
        "control": {"metric1": 0, "metric2": 0},
        "enabled": {"metric1": -2, "metric2": +1},
    },  # Experiment 2 impact
]

# Interactions
# Only metric1 has interactions.
# The interactions will push up impact of (v1, enabled)
# and we expect it to change the results due to univar effect for metric1
interaction_metric_impacts = {
    ("v1", "enabled"): {"metric1": 15},
    ("v2", "control"): {"metric1": -2},
}


def get_expected_delta(
        non_trigger_pct_multi: list[float],
        expt_metric_impacts,
        interaction_metric_impacts,
        variant,
        metric) -> float:
    """This is a simple helper function to calculate expected delta based on the
    population parameters for the simple case with two experiments."""
    # Get trigger rates
    trigger_rates = [(100.0 - x) / 100.0 for x in non_trigger_pct_multi]
    # Trigger rate for Expt 1
    trigger_rate1 = trigger_rates[0]
    # Trigger rate for Expt 2
    trigger_rate2 = trigger_rates[1]
    # Both trigger rate (both of them trigger)
    trigger_rate_both = trigger_rates[0] * trigger_rates[1]
    # Only Expt 1 triggers
    trigger_rate_only1 = trigger_rate1 - trigger_rate_both
    # Only Expt 2 triggers
    trigger_rate_only2 = trigger_rate2 - trigger_rate_both

    # Univariate impact of variant in Expt 1
    impact1 = expt_metric_impacts[0][variant[0]][metric]

    # Univariate impact of variant in Expt 2
    impact2 = expt_metric_impacts[1][variant[1]][metric]

    # Extra impact on the common trigger population
    impact_both = interaction_metric_impacts[variant][metric]

    # We need to still add univariate impacts to get the final impact on both
    impact_both += (impact1 + impact2)

    # Expected Delta
    expected_delta = (
        (trigger_rate_only1 * impact1 + trigger_rate_both * (impact_both) + trigger_rate_only2 * impact2) /
        (trigger_rate_only1 + trigger_rate_both + trigger_rate_only2)
    )

    print(expected_delta)
    return expected_delta


def simulate_and_run_mea(
        population_size=1000,
        population_seed=42,
        non_trigger_pct_multi=[5, 5],
        expt_assignment_seed_multi=[13, 17],
        control_launch=None,
        html_file_name=None):
    """
    Performing a simulation and applying MEA.

    Args:
    population_size: size of the population
    population_seed: a seed to generate a heterogenous population
    non_trigger_pct_multi: a list of floats to represent the non-triggered percent for each experiment
    control_launch: This is the "Launch" we compare various combinations with.
        By default, this will be None which maps to `Launch(value=("control", "control"))`.
        This default translates to a world where none of the experiments are ramped.
        For scenario-based analysis we can set this to be a different value.
        For example: `Launch(value=("v1", "enabled"))` is useful to understand the impact of
        Expt 2, if we assumed Expt 1: v1 is ramped to 100 percent.
    html_file_name: file name to write the report
    """
    sim = Sim(
        population_size=population_size,
        attribute_weights=attribute_weights,
        metric_attribute_values=metric_attribute_values,
        expt_variant_weights_multi=expt_variant_weights_multi,
        population_pcnt_multi=[100, 100],
        non_trigger_pct_multi=non_trigger_pct_multi,
        population_seed=population_seed,
        expt_assignment_seed_multi=expt_assignment_seed_multi,
        expt_metric_impacts=expt_metric_impacts,
        interaction_metric_impacts=interaction_metric_impacts,
    )

    sim.run()

    assert len(sim.population_df) == population_size
    assert sim.expt_df.shape[1] == 7
    assert sim.expt_metric_df.shape[1] == 9

    # Let us also create an MEA set up:
    expt1 = ExptInfo(
        test_key=None,
        experiment_id=None,
        segment_id=None,
        start_date=None,
        end_date=None,
        variants=None,
        control_label=None,
    )

    expt2 = ExptInfo(
        test_key=None,
        experiment_id=None,
        segment_id=None,
        start_date=None,
        end_date=None,
        variants=None,
        control_label=None,
    )

    metrics = [Metric(numerator=UMetric(col="metric1")), Metric(numerator=UMetric(col="metric2"))]

    analysis_info = AnalysisInfo(
        MultiExptInfo(expt_info_list=[expt1, expt2], merge_method="cross"),
        metric_info_list=[MetricInfo(metrics=metrics)],
    )


    # Let us run MEA
    mea = MEA(df=sim.expt_metric_df, analysis_info=analysis_info)
    mea.run()


    write_path = os.path.expanduser("~/mea_results/")
    os.makedirs(write_path, exist_ok=True)

    mea_report = mea.publish(
        write_path=write_path,
        add_timestamp_to_path=False,
        rounding_digits=2,
        html_file_name=html_file_name,
        markdown_file_name=None,
    )

    html_str = mea_report["html_str"]
    markdown_str = mea_report["markdown_str"]

    # Get the file names for the stored data
    # print(mea_report["file_names"])

    # You can extract the results in a dataframe format as well
    metric1_res = mea.result.metric_result_dict["metric1"].variant_effect_df_pairs
    if "(v1, enabled) launch" not in metric1_res["comparison_pair"].values:
        raise ValueError(
            "Your population was too small to have any observation of (v1, enabled) in this simulation."
            "You cannot get any conclusions on this combination.")


    res = metric1_res[metric1_res["comparison_pair"] == "(v1, enabled) launch"]
    delta = res["delta"].values[0]
    ci = res["ci"].values[0]
    # print(round(delta, 2))
    # We get: 10.2 which is close to 9.7
    # You can increase the sample size and observe the convergence

    return({"sim": sim, "mea": mea, "mea_report": mea_report, "html_str": html_str, "delta": delta, "ci": ci})

res = simulate_and_run_mea(
    population_size=1000,
    population_seed=42,
    non_trigger_pct_multi=[5, 5],
    expt_assignment_seed_multi=[13, 17],
    html_file_name="mea_test.html")


html_str = res["html_str"]
```

- [HTML](/docs/static//test-results/mea/mea_sim1/mea_test.html)
- [PDF](/docs/static//test-results/mea/mea_sim1/mea_test.pdf)


- You can display the html also in a Python notebook:

```python
html_str = res["html_str"]
from IPython.display import display, HTML
# display(HTML(html_str))
```

- Next let's run a scenario based analysis to see the impact
- In this scenario we, assess the imapct of Expt2: enabled if Expt1: v1 is launched.


```python
non_trigger_pct_multi = [5, 5]
res = simulate_and_run_mea(
    population_size=10000,
    population_seed=42,
    non_trigger_pct_multi=non_trigger_pct_multi,
    expt_assignment_seed_multi=[13, 17],
    control_launch=Launch(value=("v1", "control")))

delta = res["delta"]
print(f"delta: {delta}")
html_str = res["html_str"]
# display(HTML(html_str))
```

- Next we demonstrate the impact of lower overlap of the two experiments
- With our simulations, we can achieve a low overlap by setting

```python
non_trigger_pct_multi = [90, 90]
```

- Since the experiments are orthogonal, this will imply that:
    - `0.1 * 0.1 = 0.01` impacted by both
    - (0.1 - 0.01) = 0.09 impacted by Expt 1 only
    - (0.1 - 0.01) = 0.09 impacted by Expt 2 only
    - Therefore the expected delta would be:
```python
expected_delta = (0.09 * -2 + 0.01 * (-2 - 2 + 15) + 0.09 * -2) / (0.09 + 0.01 + 0.09)
```
    - But we could simply also use our helper function we defined before:

```python
expected_delta = get_expected_delta(
    non_trigger_pct_multi=non_trigger_pct_multi,
    expt_metric_impacts=expt_metric_impacts,
    interaction_metric_impacts=interaction_metric_impacts,
    variant=("v1", "enabled"),
    metric="metric1")
```
    - These numbers will match and expected delta will be close to -1.3


Now let us do MEA for this case and confirm the obtained estimate is close to the population parameter.

```python
non_trigger_pct_multi = [90, 90]
res = simulate_and_run_mea(
    population_size=5000,
    population_seed=42,
    non_trigger_pct_multi=non_trigger_pct_multi,
    expt_assignment_seed_multi=[13, 17],
    control_launch=Launch(value=("v1", "control")))

delta = res["delta"]
print(f"delta: {delta}")
html_str = res["html_str"]
# display(HTML(html_str))
```


Next we perform a simulation to check if the estimated delta is unbiased.
To that end we simulate the experiment data `n = 300` times by randimizing
user attributes and user assignments to experiment arms.
Then we compare the histogram of estimated deltas and true delta. We also show
one random estimate and its confidence interval.
First we define a function to simulate the flow given a fixed input for the experiment settings
but randomizing the population seed and experiment assignments.


```python
def sim_flow(
        non_trigger_pct_multi,
        population_seed,
        expt_assignment_seed_multi
        ):
    res = simulate_and_run_mea(
        population_size=5000,
        population_seed=population_seed,
        non_trigger_pct_multi=non_trigger_pct_multi,
        expt_assignment_seed_multi=expt_assignment_seed_multi,
        control_launch=Launch(value=("v1", "control")))

    delta = res["delta"]
    print(f"delta: {delta}")
    html_str = res["html_str"]
    return {"delta": delta, "ci": res["ci"]}
```


Then we run this 300 times and compare the true delta parameter (expected delta)
with histogram of 300 estimated parameters.
We also


```python
from abvelocity.utils.hist_with_quantiles import hist_with_quantiles

non_trigger_pct_multi = [90, 90]

expected_delta = get_expected_delta(
    non_trigger_pct_multi=non_trigger_pct_multi,
    expt_metric_impacts=expt_metric_impacts,
    interaction_metric_impacts=interaction_metric_impacts,
    variant=("v1", "enabled"),
    metric="metric1")

res_list = [
    sim_flow(
        non_trigger_pct_multi=non_trigger_pct_multi,
        population_seed=(j + 1317),
        expt_assignment_seed_multi=[j + 13, j + 17],
        ) for j in range(300)]

delta_array = [res["delta"] for res in res_list]
sim_delta_lower_quantile = np.quantile(delta_array, 0.025)
sim_delta_upper_quantile = np.quantile(delta_array, 0.975)
sim_delta_quantile_range = sim_delta_upper_quantile - sim_delta_lower_quantile

ci_array = [res["ci"] for res in res_list]
ci_length_array = [(ci[1] - ci[0]) for ci in ci_array]
ci_length_average = np.mean(ci_length_array)

one_sim = sim_flow(
    non_trigger_pct_multi=non_trigger_pct_multi,
    population_seed=1317,
    expt_assignment_seed_multi=[13, 17])

delta = one_sim["delta"]
ci = one_sim["ci"]
fig = hist_with_quantiles(
    x=delta_array,
    vertical_lines_dict={"one estimate": delta, "expected delta": expected_delta},
    bands=list(ci))

# The figure shows that the estimated deltas are centered around the true delta (expected).
# This indicates that our estimation is not biased.
fig.show()

# Compare the simulation based variance in delta with the average length of the estimated CI.
# We observe that these values are close showing our uncertainty estimation is reasonable.
print(f"\n***: sim_delta_quantile_range: {sim_delta_quantile_range}")
print(f"\n***: ci_length_average: {ci_length_average}")
```

![Histogram of Simulation Data](sim.png)

