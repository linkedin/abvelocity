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
"""The goal is to simulate experiment data."""

import numpy as np
from abvelocity.core.get_data.data_container import DataContainer
from abvelocity.core.get_data.get_expt_stats import get_expt_stats
from abvelocity.core.param.constants import CATEG_NAN_VALUE, TRIGGER_STATE_COUNT_COL
from abvelocity.core.param.variant import ComparisonPair, Variant, VariantList
from abvelocity.core.sim.sim import Sim
from abvelocity.core.stats.param import StrataInfo

POPULATION_SIZE = 20000
SEED = 1317


# Simulation constants
ATTRIBUTE_WEIGHTS = {
    "Level": {"Senior": 0.5, "Junior": 0.5},
    "Country": {"Iran": 0.1, "Canada": 0.3, "UK": 0.2, "US": 0.4},
    "Device": {"Smartphone": 0.6, "Tablet": 0.3, "Laptop": 0.1},
}
METRIC_ATTRIBUTE_VALUES = {
    "impressions": {"Level": {"Junior": 100, "Senior": 90}},
    "clicks": {"Level": {"Junior": 10, "Senior": 8}},
}
EXPT_VARIANT_WEIGHTS_MULTI = [
    {"control": 0.5, "enabled": 0.5},
    {"control": 0.5, "v1": 0.5},
]
POPULATION_PCNT_MULTI = [100, 100]
NON_TRIGGER_PCT_MULTI = [10, 10]
EXPT_METRIC_IMPACTS = [
    {"control": {"impressions": 0, "clicks": 0}, "enabled": {"impressions": 10, "clicks": 2}},
    {"control": {"impressions": 0, "clicks": 0}, "v1": {"impressions": 5, "clicks": 1}},
]

LAUNCH_VALUE = ("enabled", "v1")
# Comparison pair constants
CONTROL_VARIANT_VALUES = [
    ("control", CATEG_NAN_VALUE),
    (CATEG_NAN_VALUE, "control"),
    ("control", "control"),
]
TREATMENT_VARIANT_VALUES = [
    ("enabled", CATEG_NAN_VALUE),
    (CATEG_NAN_VALUE, "v1"),
    ("enabled", "v1"),
]

INTERACTION_METRIC_IMPACTS = {LAUNCH_VALUE: {"impressions": 15, "clicks": 3}}


COMPARISON_PAIR = ComparisonPair(
    control=VariantList(variants=[Variant(value=v) for v in CONTROL_VARIANT_VALUES]),
    treatment=VariantList(variants=[Variant(value=v) for v in TREATMENT_VARIANT_VALUES]),
    name=f"sim focus: launch {LAUNCH_VALUE}",
)

COMPARISON_PAIRS = [COMPARISON_PAIR]


# Weighted mean function
def weighted_mean(df, metric_col, variant_values, strata_info):
    """Compute weighted mean for a metric using StrataInfo weights.

    Args:
        df (pd.DataFrame): The DataFrame containing experiment metric data.
        metric_col (str): The name of the metric column (e.g., 'clicks').
        variant_values (list[tuple]): A list of variant tuples to include in the weighted mean.
        strata_info (StrataInfo): Strata information containing variant counts.

    Returns:
        float: The weighted mean of the metric across the specified variants.
    """
    counts = [strata_info.df.at[v, TRIGGER_STATE_COUNT_COL] if v in strata_info.df.index else 0 for v in variant_values]
    total_count = sum(counts)
    weights = [c / total_count if total_count > 0 else 0 for c in counts]
    total = 0.0
    for v, w in zip(variant_values, weights):
        if v in strata_info.df.index:
            variant_data = df[df["variant"] == v][metric_col]
            # Use iloc[0] to get the scalar value from the series mean
            variant_mean = variant_data.mean() if len(variant_data) > 0 else 0
            total += w * variant_mean
    return total


def calc_expected_delta(
    non_trigger_pct_multi: list[float],
    expt_metric_impacts,
    interaction_metric_impacts,
    launch_value: tuple,
    metric: str,
) -> float:
    """Calculate expected delta based on population parameters.

    Args:
        non_trigger_pct_multi (list[float]): List of non-trigger percentages (e.g., [10, 10]).
        expt_metric_impacts (list[dict]): List of dictionaries defining metric impacts for each experiment.
        interaction_metric_impacts (dict): Dictionary defining metric impacts for variant interactions.
        launch_value (tuple): The specific variant combination being analyzed (e.g., ('enabled', 'v1')).
        metric (str): The name of the metric (e.g., 'impressions').

    Returns:
        float: The expected delta (mean change) of the metric for the given launch value.
    """
    # Convert non-trigger percentage (e.g., 10%) to trigger rate (e.g., 0.9)
    trigger_rates = [(100.0 - x) / 100.0 for x in non_trigger_pct_multi]
    trigger_rate1 = trigger_rates[0]
    trigger_rate2 = trigger_rates[1]

    # Calculate probabilities for mutually exclusive trigger states
    trigger_rate_both = trigger_rate1 * trigger_rate2
    trigger_rate_only1 = trigger_rate1 - trigger_rate_both
    trigger_rate_only2 = trigger_rate2 - trigger_rate_both

    # Get the raw impact values
    impact1 = expt_metric_impacts[0][launch_value[0]][metric]
    impact2 = expt_metric_impacts[1][launch_value[1]][metric]

    # Calculate impact when both trigger
    impact_both = interaction_metric_impacts.get(launch_value, {}).get(metric, 0)
    impact_both += impact1 + impact2

    # The total proportion of the population that triggers at least one experiment
    trigger_rate_sum = trigger_rate_only1 + trigger_rate_both + trigger_rate_only2

    # Assert checks for internal consistency of trigger rates (used in the original code, kept here)
    # Note: 0.99 seems to be the expected total trigger rate based on the current inputs (0.9 + 0.9 - 0.81 = 0.99)
    assert np.isclose(trigger_rate_sum, 0.99, rtol=0.001)

    # Weighted average of impacts across triggering strata
    expected_delta = (trigger_rate_only1 * impact1 + trigger_rate_both * impact_both + trigger_rate_only2 * impact2) / trigger_rate_sum
    return expected_delta


def sim_ctr_data(population_size: int = POPULATION_SIZE, seed: int = SEED):
    """Creates DataFrame, StrataInfo, and true values from a single Sim run.

    Args:
        population_size (int): The number of samples to generate in the simulation. Defaults to POPULATION_SIZE (20000).
        seed (int): The random seed used for population and assignment generation. Defaults to SEED (1317).

    Returns:
        tuple: A tuple containing:
            - df (pd.DataFrame): The simulated experiment metric data.
            - strata_info (StrataInfo): Strata information based on variant counts.
            - true_values (dict): Dictionary of true/expected metric differences:
                'impressions_diff' (float): Absolute difference in impressions mean.
                'impressions_pct' (float): Percentage difference in impressions mean.
                'ctr_diff' (float): Absolute difference in CTR.
                'ctr_pct' (float): Percentage difference in CTR.
    """
    sim = Sim(
        population_size=population_size,
        attribute_weights=ATTRIBUTE_WEIGHTS,
        metric_attribute_values=METRIC_ATTRIBUTE_VALUES,
        expt_variant_weights_multi=EXPT_VARIANT_WEIGHTS_MULTI,
        population_pcnt_multi=POPULATION_PCNT_MULTI,
        non_trigger_pct_multi=NON_TRIGGER_PCT_MULTI,
        population_seed=seed,
        expt_assignment_seed_multi=[seed, seed + 1],
        expt_metric_impacts=EXPT_METRIC_IMPACTS,
        interaction_metric_impacts=INTERACTION_METRIC_IMPACTS,
    )
    sim.run()
    df = sim.expt_metric_df.copy()
    # Shuffle the DataFrame to simulate real-world data order
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Get experiment statistics to derive StrataInfo
    expt_stats = get_expt_stats(DataContainer(pandas_df=df))
    strata_df = expt_stats.variant_count_df[[TRIGGER_STATE_COUNT_COL]].copy()
    strata_df["variant"] = strata_df[TRIGGER_STATE_COUNT_COL]
    strata_info = StrataInfo(df=strata_df, strata_count_col=TRIGGER_STATE_COUNT_COL)

    # Define variants for calculating expected/true values
    launch_value = LAUNCH_VALUE
    control_value = ("control", "control")

    # Calculate true impressions difference (absolute and percentage)
    impressions_t = calc_expected_delta(
        NON_TRIGGER_PCT_MULTI,
        EXPT_METRIC_IMPACTS,
        INTERACTION_METRIC_IMPACTS,
        launch_value,
        "impressions",
    )
    impressions_c = calc_expected_delta(
        NON_TRIGGER_PCT_MULTI,
        EXPT_METRIC_IMPACTS,
        INTERACTION_METRIC_IMPACTS,
        control_value,
        "impressions",
    )
    true_impressions_diff = impressions_t - impressions_c
    # 95 is likely the baseline average impression value for the control group
    true_impressions_pct_diff = 100 * (true_impressions_diff / 95)

    # Calculate true CTR difference (absolute) using weighted mean
    ctr_t = weighted_mean(df, "clicks", TREATMENT_VARIANT_VALUES, strata_info) / weighted_mean(df, "impressions", TREATMENT_VARIANT_VALUES, strata_info)

    ctr_c = weighted_mean(df, "clicks", CONTROL_VARIANT_VALUES, strata_info) / weighted_mean(df, "impressions", CONTROL_VARIANT_VALUES, strata_info)

    true_ctr_diff = ctr_t - ctr_c

    # Calculate true CTR difference (percentage)
    true_ctr_pct = true_ctr_diff / ctr_c * 100

    return (
        df,
        strata_info,
        {
            "impressions_diff": true_impressions_diff,
            "impressions_pct_diff": true_impressions_pct_diff,
            "ctr_diff": true_ctr_diff,
            "ctr_pct_diff": true_ctr_pct,
        },
    )
