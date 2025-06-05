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

from abvelocity.sim.sim import Sim

ATTRIBUTE_WEIGHTS = {
    "Level": {"Senior": 0.5, "Junior": 0.5},
    "Country": {"Iran": 0.1, "Canada": 0.3, "UK": 0.2, "US": 0.4},
    "Device": {"Smartphone": 0.6, "Tablet": 0.3, "Laptop": 0.1},
}
"""Example Define weights for attributes of the population."""


METRIC_ATTRIBUT_VALUES = {
    "metric1": {"Level": {"Junior": 5, "Senior": 2}, "Country": {"Iran": +5}},
    "metric2": {"Country": {"USA": 5, "Canada": 2}},
}
"""Example baseline metrics which depends on the attributes (but not experiment assignments)."""


POPULATION_SIZE = 1000
"""Default population size."""


POPULATION_SEED = 42
"""Default population seed."""


def simulate_data_multi1(
    population_size=POPULATION_SIZE,
    population_seed=POPULATION_SEED,
    expt_assignment_seed_multi=tuple([13, 17]),
    population_pcnt_multi=tuple([100, 100]),
    non_trigger_pct_multi=tuple([5, 5]),
) -> type(Sim):
    """
    This function creates simulated data to be used in tests.
    This has expt effects for two experiments on two metrics.
    But the interaction effect is only on metric1.
    """
    # Example parameters for multiple experiment assignment weights
    expt_variant_weights_multi = [
        {"control": 0.4, "v1": 0.3, "v2": 0.3},  # Experiment 1
        {"control": 0.3, "enabled": 0.7},  # Experiment 2
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

    sim = Sim(
        population_size=population_size,
        attribute_weights=ATTRIBUTE_WEIGHTS,
        metric_attribute_values=METRIC_ATTRIBUT_VALUES,
        expt_variant_weights_multi=expt_variant_weights_multi,
        population_pcnt_multi=population_pcnt_multi,
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

    return sim


def simulate_data_multi2(
    population_size=POPULATION_SIZE,
    population_seed=POPULATION_SEED,
    expt_assignment_seed_multi=tuple([13, 17]),
    population_pcnt_multi=tuple([90, 90]),
    non_trigger_pct_multi=tuple([5, 5]),
) -> type(Sim):
    """
    This function creates simulated data to be used in tests.
    This has only experiment and interaction effects for metric1.
    """
    # Example parameters for multiple experiment assignment weights
    expt_variant_weights_multi = [
        {"control": 0.4, "v1": 0.3, "v2": 0.3},  # Experiment 1
        {"control": 0.3, "enabled": 0.7},  # Experiment 2
    ]

    # Univariate Effects
    # For metric1: v1 is bad; enabled is bad. v2 is good.
    # For metric2: no impact
    expt_metric_impacts = [
        {
            "control": {"metric1": 1.5},
            "v1": {"metric1": -2, "metric2": 0},
            "v2": {"metric1": +5},
        },  # Experiment 1 impact
        {
            "control": {"metric2": 0},
            "enabled": {"metric1": -2, "metric2": 0},
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

    sim = Sim(
        population_size=population_size,
        attribute_weights=ATTRIBUTE_WEIGHTS,
        metric_attribute_values=METRIC_ATTRIBUT_VALUES,
        expt_variant_weights_multi=expt_variant_weights_multi,
        population_pcnt_multi=population_pcnt_multi,
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

    return sim


def simulate_data_multi3(
    population_size=POPULATION_SIZE,
    population_seed=POPULATION_SEED,
    expt_assignment_seed_multi=tuple([13, 17]),
    population_pcnt_multi=tuple([90, 90]),
    non_trigger_pct_multi=tuple([5, 5]),
) -> type(Sim):
    """
    This function creates simulated data to be used in tests.
    In this case the effects come only through interactions and on metric1.
    """
    # Example parameters for multiple experiment assignment weights
    expt_variant_weights_multi = [
        {"control": 0.4, "v1": 0.3, "v2": 0.3},  # Experiment 1
        {"control": 0.3, "enabled": 0.7},  # Experiment 2
    ]

    # No Univariate Effects
    # For metric1: v1 is bad; enabled is bad. v2 is good.
    # For metric2: no impact
    expt_metric_impacts = None

    # Interactions
    # Only metric1 has interactions.
    # The interactions will push up impact of (v1, enabled)
    # and we expect it to change the results due to univar effect for metric1
    interaction_metric_impacts = {
        ("v1", "enabled"): {"metric1": 15},
        ("v2", "control"): {"metric1": -2},
    }

    sim = Sim(
        population_size=population_size,
        attribute_weights=ATTRIBUTE_WEIGHTS,
        metric_attribute_values=METRIC_ATTRIBUT_VALUES,
        expt_variant_weights_multi=expt_variant_weights_multi,
        population_pcnt_multi=population_pcnt_multi,
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
    return sim


def simulate_data_multi4(
    population_size=POPULATION_SIZE,
    population_seed=POPULATION_SEED,
    expt_assignment_seed_multi=tuple([13, 17]),
    population_pcnt_multi=tuple([90, 90]),
    non_trigger_pct_multi=tuple([5, 5]),
) -> type(Sim):
    """
    This function creates simulated data to be used in tests.
    In this case the experiment assignments in each experiment are even.
    """
    # Example parameters for multiple experiment assignment weights
    expt_variant_weights_multi = [
        {"control": 1.0 / 3.0, "v1": 1.0 / 3.0, "v2": 1.0 / 3.0},  # Experiment 1
        {"control": 0.5, "enabled": 0.5},  # Experiment 2
    ]

    # No Univariate Effects
    # For metric1: v1 is bad; enabled is bad. v2 is good.
    # For metric2: no impact
    expt_metric_impacts = None

    # Interactions
    # Only metric1 has interactions.
    # The interactions will push up impact of (v1, enabled)
    # and we expect it to change the results due to univar effect for metric1
    interaction_metric_impacts = {
        ("v1", "enabled"): {"metric1": 15},
        ("v2", "control"): {"metric1": -2},
    }

    sim = Sim(
        population_size=population_size,
        attribute_weights=ATTRIBUTE_WEIGHTS,
        metric_attribute_values=METRIC_ATTRIBUT_VALUES,
        expt_variant_weights_multi=expt_variant_weights_multi,
        population_pcnt_multi=population_pcnt_multi,
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
    return sim


def simulate_data_multi5(
    population_size=POPULATION_SIZE,
    population_seed=POPULATION_SEED,
    expt_assignment_seed_multi=tuple([13, 17]),
    population_pcnt_multi=tuple([90, 90]),
    non_trigger_pct_multi=tuple([5, 5]),
) -> type(Sim):
    """
    This function creates simulated data to be used in tests.
    In this case the experiment assignments in each experiment are even and each
    only have two arms 50/50 split.
    """
    # Example parameters for multiple experiment assignment weights
    expt_variant_weights_multi = [
        {"control": 0.5, "v1": 0.5},  # Experiment 1
        {"control": 0.5, "enabled": 0.5},  # Experiment 2
    ]

    # No Univariate Effects
    # For metric1: v1 is bad; enabled is bad. v2 is good.
    # For metric2: no impact
    expt_metric_impacts = None

    # Interactions
    # Only metric1 has interactions.
    # The interactions will push up impact of (v1, enabled)
    # and we expect it to change the results due to univar effect for metric1
    interaction_metric_impacts = {("v1", "enabled"): {"metric1": 15}}

    sim = Sim(
        population_size=population_size,
        attribute_weights=ATTRIBUTE_WEIGHTS,
        metric_attribute_values=METRIC_ATTRIBUT_VALUES,
        expt_variant_weights_multi=expt_variant_weights_multi,
        population_pcnt_multi=population_pcnt_multi,
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
    return sim


def simulate_data_uni1(
    population_size=POPULATION_SIZE,
    population_seed=POPULATION_SEED,
    expt_assignment_seed_multi=tuple([13]),
    population_pcnt_multi=tuple([90]),
    non_trigger_pct_multi=tuple([5]),
) -> type(Sim):
    """
    An example with one experiment only.
    Only metric1 is impacted.
    """

    # Example parameters for univar experiment assignment weights
    # The assignment weights differ across variants.
    expt_variant_weights_multi = [
        {"control": 0.4, "v1": 0.3, "v2": 0.3},  # Experiment 1 (the only experiment)
    ]

    # Define metrics impacts
    # Only metric 1 is impacted
    expt_metric_impacts = [
        {"control": {"metric1": 1.5}, "v1": {"metric1": -2, "metric2": 0}, "v2": {"metric1": +5}},
    ]

    sim = Sim(
        population_size=population_size,
        attribute_weights=ATTRIBUTE_WEIGHTS,
        metric_attribute_values=METRIC_ATTRIBUT_VALUES,
        expt_variant_weights_multi=expt_variant_weights_multi,
        population_pcnt_multi=population_pcnt_multi,
        non_trigger_pct_multi=non_trigger_pct_multi,
        population_seed=population_seed,
        expt_assignment_seed_multi=expt_assignment_seed_multi,
        expt_metric_impacts=expt_metric_impacts,
        interaction_metric_impacts=None,
    )

    sim.run()

    assert len(sim.population_df) == population_size
    assert sim.expt_df.shape[1] == 6
    assert sim.expt_metric_df.shape[1] == 8

    return sim


def simulate_data_uni2(
    population_size=POPULATION_SIZE,
    population_seed=POPULATION_SEED,
    expt_assignment_seed_multi=tuple([13]),
    population_pcnt_multi=tuple([90]),
    non_trigger_pct_multi=tuple([5]),
) -> type(Sim):
    """
    An example with one experiment only.
    Only metric1 is impacted.
    The population is split equally for three variants.
    """

    # Example parameters for univar experiment assignment weights
    # This time assignment weights are equal among the three variants.
    expt_variant_weights_multi = [
        {
            "control": 1.0 / 3.0,
            "v1": 1.0 / 3.0,
            "v2": 1.0 / 3.0,
        },  # Experiment 1 (the only experiment)
    ]

    # Define metrics impacts
    # Only metric 1 is impacted
    expt_metric_impacts = [
        {"control": {"metric1": 1.5}, "v1": {"metric1": -2}, "v2": {"metric1": +5}},
    ]

    sim = Sim(
        population_size=population_size,
        attribute_weights=ATTRIBUTE_WEIGHTS,
        metric_attribute_values=METRIC_ATTRIBUT_VALUES,
        expt_variant_weights_multi=expt_variant_weights_multi,
        population_pcnt_multi=population_pcnt_multi,
        non_trigger_pct_multi=non_trigger_pct_multi,
        population_seed=population_seed,
        expt_assignment_seed_multi=expt_assignment_seed_multi,
        expt_metric_impacts=expt_metric_impacts,
        interaction_metric_impacts=None,
    )

    sim.run()

    assert len(sim.population_df) == population_size
    assert sim.expt_df.shape[1] == 6
    assert sim.expt_metric_df.shape[1] == 8

    return sim


def simulate_data_uni3(
    population_size=POPULATION_SIZE,
    population_seed=POPULATION_SEED,
    expt_assignment_seed_multi=tuple([13]),
    population_pcnt_multi=tuple([90]),
    non_trigger_pct_multi=tuple([5]),
) -> type(Sim):
    """
    An example with one experiment only.
    Only metric1 is impacted.
    The population is split equally for only two variants (control and enabled).
    """

    # Example parameters for univar experiment assignment weights
    # This time assignment weights are equal among the three variants.
    expt_variant_weights_multi = [
        {"control": 1.0 / 2.0, "enabled": 1.0 / 2.0},  # Experiment 1 (the only experiment)
    ]

    # Define metrics impacts
    # Only metric 1 is impacted
    expt_metric_impacts = [
        {"control": {"metric1": 1.5}, "enabled": {"metric1": -2}},
    ]

    sim = Sim(
        population_size=population_size,
        attribute_weights=ATTRIBUTE_WEIGHTS,
        metric_attribute_values=METRIC_ATTRIBUT_VALUES,
        expt_variant_weights_multi=expt_variant_weights_multi,
        population_pcnt_multi=population_pcnt_multi,
        non_trigger_pct_multi=non_trigger_pct_multi,
        population_seed=population_seed,
        expt_assignment_seed_multi=expt_assignment_seed_multi,
        expt_metric_impacts=expt_metric_impacts,
        interaction_metric_impacts=None,
    )

    sim.run()

    assert len(sim.population_df) == population_size
    assert sim.expt_df.shape[1] == 6
    assert sim.expt_metric_df.shape[1] == 8

    return sim


def simulate_data_three1(
    population_size=POPULATION_SIZE,
    population_seed=POPULATION_SEED,
    expt_assignment_seed_multi=tuple([13, 17, 19]),
    population_pcnt_multi=tuple([90, 90, 90]),
    non_trigger_pct_multi=tuple([5, 5, 5]),
) -> type(Sim):
    """
    This function creates simulated data to be used in tests.
    This has expt effects for three experiments on two metrics.
    But the interaction effect is only on metric1.
    """
    # Example parameters for multiple experiment assignment weights
    expt_variant_weights_multi = [
        {"control": 0.4, "v1": 0.3, "v2": 0.3},  # Experiment 1
        {"control": 0.3, "enabled": 0.7},  # Experiment 2
        {"control": 0.3, "enabled": 0.7},  # Experiment 3
    ]

    # Univariate Effects
    # For metric1: v1 is bad; enabled is bad. v2 is good.
    # For metric2: v2 is good. control is good.
    expt_metric_impacts = [
        {
            "control": {"metric1": 1.5},
            "v1": {"metric1": -2, "metric2": +5},
            "v2": {"metric1": +5},
        },  # Experiment 1 impact
        {
            "control": {"metric2": 1},
            "enabled": {"metric1": -2, "metric2": -5},
        },  # Experiment 2 impact
        {},  # Experiment 3, no impact
    ]

    # Interactions
    # Only metric1 has interactions.
    # The interactions will push up impact of (v1, enabled)
    # and we expect it to change the results due to univar effect for metric1
    interaction_metric_impacts = {
        ("v1", "enabled", "enabled"): {"metric1": 25},
        ("v2", "control", "control"): {"metric1": -2},
    }

    sim = Sim(
        population_size=population_size,
        attribute_weights=ATTRIBUTE_WEIGHTS,
        metric_attribute_values=METRIC_ATTRIBUT_VALUES,
        expt_variant_weights_multi=expt_variant_weights_multi,
        population_pcnt_multi=population_pcnt_multi,
        non_trigger_pct_multi=non_trigger_pct_multi,
        population_seed=population_seed,
        expt_assignment_seed_multi=expt_assignment_seed_multi,
        expt_metric_impacts=expt_metric_impacts,
        interaction_metric_impacts=interaction_metric_impacts,
    )

    sim.run()

    assert len(sim.population_df) == population_size
    assert sim.expt_df.shape[1] == 8
    assert sim.expt_metric_df.shape[1] == 10

    return sim


def simulate_data_three2(
    population_size=POPULATION_SIZE,
    population_seed=POPULATION_SEED,
    expt_assignment_seed_multi=tuple([13, 17, 19]),
    population_pcnt_multi=tuple([90, 90, 90]),
    non_trigger_pct_multi=tuple([5, 5, 5]),
) -> type(Sim):
    """
    This function creates simulated data to be used in tests.
    This has expt effects for three experiments on two metrics.
    But the interaction effect is only on metric1.
    """
    # Example parameters for multiple experiment assignment weights
    expt_variant_weights_multi = [
        {"control": 0.4, "v1": 0.3, "v2": 0.3},  # Experiment 1
        {"control": 0.3, "enabled": 0.7},  # Experiment 2
        {"control": 0.3, "enabled": 0.7},  # Experiment 3
    ]

    # Univariate Effects
    # For metric1: v1 is bad; enabled is bad. v2 is good.
    # For metric2: v2 is good. control is good.
    expt_metric_impacts = [
        {
            "control": {"metric1": 1.5},
            "v1": {"metric1": -2, "metric2": 0},
            "v2": {"metric1": +5},
        },  # Experiment 1 impact
        {
            "control": {"metric2": 0},
            "enabled": {"metric1": -2, "metric2": 0},
        },  # Experiment 2 impact
        {},  # Experiment 3, no impact
    ]

    # Interactions
    # Only metric1 has interactions.
    # The interactions will push up impact of (v1, enabled)
    # and we expect it to change the results due to univar effect for metric1
    interaction_metric_impacts = {
        ("v1", "enabled", "enabled"): {"metric1": 25},
        ("v2", "control", "control"): {"metric1": -2},
    }

    sim = Sim(
        population_size=population_size,
        attribute_weights=ATTRIBUTE_WEIGHTS,
        metric_attribute_values=METRIC_ATTRIBUT_VALUES,
        expt_variant_weights_multi=expt_variant_weights_multi,
        population_pcnt_multi=population_pcnt_multi,
        non_trigger_pct_multi=non_trigger_pct_multi,
        population_seed=population_seed,
        expt_assignment_seed_multi=expt_assignment_seed_multi,
        expt_metric_impacts=expt_metric_impacts,
        interaction_metric_impacts=interaction_metric_impacts,
    )

    sim.run()

    assert len(sim.population_df) == population_size
    assert sim.expt_df.shape[1] == 8
    assert sim.expt_metric_df.shape[1] == 10

    return sim
