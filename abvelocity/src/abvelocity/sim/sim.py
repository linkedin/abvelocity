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

import copy
from typing import Optional, TextIO

import numpy as np
import pandas as pd

from abvelocity.get_data.data_container import DataContainer
from abvelocity.get_data.join_expt_dfs import join_expt_dfs
from abvelocity.param.constants import CATEG_NAN_VALUE, UNIT_COL, VARIANT_COL


class Sim:
    """
    The goal is to simulate experiment data.

        - it allows to introduce attributes e.g. country etc in the population
        - it allows for overlapping multiple experiments
        - it allows to add baseline metrics which depend on the attributes (but not the experiment assignments)
        - it allows for adding univariate experiment impacts
        - it allows for adding interaction effects (multi-experiment)

    # Here is an example simulation

    # Population attributes
    attribute_weights = {
        "Level": {"Senior": 0.5, "Junior": 0.5},
        "Country": {"Iran": 0.1, "Canada": 0.3, "UK": 0.2, "US": 0.4},
        "Device": {"Smartphone": 0.6, "Tablet": 0.3, "Laptop": 0.1},
    }

    # Example parameters for multiple experiments
    expt_variant_weights_multi = [
        {"control": 0.4, "v1": 0.3, "v2": 0.3},  # Experiment 1
        {"control": 0.3, "enabled": 0.7},  # Experiment 2
    ]

    # Baseline metrics which depends on attributes (no expt is applied yet)
    metric_attribute_values = {
        "metric1": {"Level": {"Junior": 10, "Senior": 5}},
        "metric2": {"Country": {"USA": 20, "Canada": 15}},
    }

    # Univariate Effects
    # For metric1: v1 is bad; enabled is bad. v2 is good.
    # For metric2: v2 is good. control is good.
    expt_metric_impacts = [
        {
            "control": {"metric1": 1.5},
            "v1": {"metric1": -2, "metric2": +5},
            "v2": {"metric1": +5},
        },  # Experiment 1 impact
        {"control": {"metric2": 1}, "enabled": {"metric1": -2, "metric2": -5}},  # Experiment 2 impact
    ]

    # Interactions
    # Only metric1 has interactions.
    # The interactions will push up impact of (v1, enabled)
    # and we expect it to change the results due to univar effect for metric1

    interaction_metric_impacts = {("v1", "enabled"): {"metric1": 15}, ("v2", "control"): {"metric1": -2}}

    sim = Sim(
        population_size=1000,
        attribute_weights=attribute_weights,
        metric_attribute_values=metric_attribute_values,
        expt_variant_weights_multi=expt_variant_weights_multi,
        population_pcnt_multi=[90, 90],
        non_trigger_pct_multi=[5, 5],
        population_seed=42,
        expt_assignment_seed_multi=[13, 17],
        expt_metric_impacts=expt_metric_impacts,
        interaction_metric_impacts=interaction_metric_impacts,
    )

    sim.run()

    assert len(sim.population_df) == 1000

    for expt_data in sim.expt_data_list:
        expt_df = expt_data[0]
        assert len(expt_df) == 900

    assert sim.expt_df.shape == (985, 7)
    assert sim.expt_metric_df.shape == (985, 9)

    # Get a summary of how the data looks like.
    sim.publish(file=None)

    """

    def __init__(
        self,
        population_size: int = 0,
        attribute_weights: dict[str, dict[str, float]] = dict(),
        metric_attribute_values: dict[str, dict[str, dict[str, float]]] = dict(),
        expt_variant_weights_multi: tuple[dict[str, float]] = tuple(),
        population_pcnt_multi: tuple[float] = tuple(),
        non_trigger_pct_multi: tuple[float] = tuple(),
        population_seed: Optional[int] = None,
        expt_assignment_seed_multi: tuple[int] = tuple(),
        expt_metric_impacts: tuple[dict[str, float]] = tuple(),
        interaction_metric_impacts: dict[tuple[str], dict[str, float]] = dict(),
    ):
        # Based on input
        self.population_size = population_size
        self.attribute_weights = attribute_weights
        self.metric_attribute_values = metric_attribute_values
        self.expt_variant_weights_multi = expt_variant_weights_multi
        self.population_pcnt_multi = population_pcnt_multi
        self.non_trigger_pct_multi = non_trigger_pct_multi
        self.population_seed = population_seed
        self.expt_assignment_seed_multi = expt_assignment_seed_multi
        self.expt_metric_impacts = expt_metric_impacts
        self.interaction_metric_impacts = interaction_metric_impacts

        # Attributes generated by the `run` method
        # Population dataframe
        self.population_df = None
        self.population_summary = None

        # A list of tuples with first element being the expt dataframe
        # second element: summary of that data
        self.expt_data_list = None

        # The join experiment assignment data (no metrics)
        self.expt_df = None
        self.expt_df_summary = None

        # Baseline metric data
        self.expt_metric_baseline_df = None

        # Joined data of experiments and
        self.expt_metric_df = None

    def generate_population(
        self,
        population_size: int,
        attribute_weights: dict[str, dict[str, float]],
        seed: Optional[int] = None,
    ) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
        """
        Generates a population DataFrame based on attribute weights and computes the attribute proportions.

        Args:
            population_size: The size of the population to generate.
            attribute_weights: A dictionary where keys are attribute names and values
                are dictionaries of attribute values with their corresponding probabilities.
            seed: An optional seed for the random number generator. Defaults to None.

        Returns:
            result: A tuple containing the population DataFrame and
                a dictionary with attribute proportions.
        """
        rng = np.random.default_rng(seed)

        data = {UNIT_COL: list(range(1, population_size + 1))}

        # Generate distributions for each attribute based on weights
        for attribute, weights in attribute_weights.items():
            choices = list(weights.keys())
            probs = list(weights.values())
            distribution = rng.choice(choices, p=probs, size=population_size)
            data[attribute] = distribution

        # Create DataFrame
        population_df = pd.DataFrame(data)

        # Compute summaries
        summary = {}
        for attribute in attribute_weights.keys():
            attribute_summary = population_df[attribute].value_counts(normalize=True).to_dict()
            summary[f"{attribute} Proportion"] = attribute_summary

        return population_df, summary

    def add_baseline_metrics(
        self, df: pd.DataFrame, metric_attribute_values: dict[str, dict[str, dict[str, float]]]
    ) -> pd.DataFrame:
        """
        Adds baseline metrics to a DataFrame based on attribute values and coefficients.

        Args:
            df: A DataFrame containing the population data.
            metric_attribute_values:
                A nested dictionary where the first-level keys are metric names, the second-level
                keys are attribute names, and the third-level keys are attribute values with their
                corresponding coefficients. Defaults to None.

        Returns:
            result: The original DataFrame with added baseline metrics.
        """
        if metric_attribute_values:
            for metric_name, coeffs in metric_attribute_values.items():
                metric_values = np.zeros(len(df))
                for attribute, attribute_coeffs in coeffs.items():
                    for value, coefficient in attribute_coeffs.items():
                        metric_values += np.where(df[attribute] == value, coefficient, 0)
                        df[metric_name] = metric_values

        return df

    def simulate_expt_assignment(
        self,
        population_df: pd.DataFrame,
        expt_variant_weights: dict[str, float],
        population_pcnt: float = 100.0,
        non_trigger_pct: float = 0.0,
        seed: Optional[int] = None,
    ) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
        """
        Simulates experiment assignment for a sample of the population.

        Args:
            population_df: A DataFrame containing the population data.
            expt_variant_weights: A dictionary where keys are experiment variants and values are their
                corresponding probabilities.
            population_pcnt: The percentage of the population to be sampled for the experiment.
            non_trigger_pct: The percentage of the sample that should not be assigned to any experiment
                variant (denoted by a specific value). Defaults to 0.0.
            seed: An optional seed for the random number generator. Defaults to None.

        Returns:
            result: A tuple containing the sampled DataFrame with experiment
                assignments and a dictionary summarizing the proportions of each variant.
        """
        rng = np.random.default_rng(seed)

        sample_size = int(len(population_df) * (population_pcnt / 100.0))
        expt_df = population_df.sample(n=sample_size, random_state=seed)

        # Perform deep copy to preserve the original input
        expt_variant_weights = copy.deepcopy(expt_variant_weights)

        if non_trigger_pct > 0.0:
            expt_variant_weights[CATEG_NAN_VALUE] = non_trigger_pct / 100.0
            # We need to make sure the weights still sum up to 1
            # To correct that we need to correct the weights for all variants except for `CATEG_NAN_VALUE`
            # We can only allocate (100 - non_trigger_pct) percent to them
            # So their original weight needs to be multiplied by `coef` defined below
            coef = (100.0 - non_trigger_pct) / 100.0
            for k in expt_variant_weights.keys():
                if k != CATEG_NAN_VALUE:
                    expt_variant_weights[k] = expt_variant_weights[k] * coef
        experiment_choices = list(expt_variant_weights.keys())
        experiment_probs = list(expt_variant_weights.values())
        expt_df[VARIANT_COL] = rng.choice(experiment_choices, p=experiment_probs, size=sample_size)

        summary = expt_df[VARIANT_COL].value_counts(normalize=True).to_dict()

        return expt_df, summary

    def simulate_expt_assignment_multi(
        self,
        population_df: pd.DataFrame,
        expt_variant_weights_multi: tuple[dict[str, float]],
        population_pcnt_multi: tuple[float] = tuple(),
        non_trigger_pct_multi: tuple[float] = tuple(),
        seed_multi: tuple[int] = tuple(),
    ) -> list[tuple[pd.DataFrame, dict[str, dict[str, float]]]]:
        """
        Samples the population for multiple experiments and returns the results.

        Args:
            population_df: The generated population DataFrame.
            expt_variant_weights_multi: List of experiment weights for each experiment.
            population_pcnt_multi: Percentage of the population to sample.
            non_trigger_pct_multi: List of non-trigger percentages for each experiment.
            seed_multi: Random seed for reproducibility.

        Returns:
            result: List of sampled DataFrames and their variant summaries.
        """
        if not expt_variant_weights_multi:
            raise ValueError("`expt_variant_weights_multi` cannot be None or empty.")

        k = len(expt_variant_weights_multi)  # Infer the number of experiments (k)
        results = []

        for i in range(k):
            expt_variant_weights = expt_variant_weights_multi[i]

            population_pcnt = 100.0
            if population_pcnt_multi:
                population_pcnt = population_pcnt_multi[i]

            non_trigger_pct = 0.0
            if non_trigger_pct_multi:
                non_trigger_pct = non_trigger_pct_multi[i]

            seed = None
            if seed_multi:
                seed = seed_multi[i]

            expt_df, variant_summary = self.simulate_expt_assignment(
                population_df=population_df,
                expt_variant_weights=expt_variant_weights,
                population_pcnt=population_pcnt,
                non_trigger_pct=non_trigger_pct,
                seed=seed,
            )
            results.append((expt_df, variant_summary))

        return results

    def merge_expt_dfs(
        self, expt_df_list: list[pd.DataFrame], attributes: tuple[str] = tuple()
    ) -> tuple[pd.DataFrame, dict]:
        """
        Merges results from multiple experiments into a single DataFrame with variant tuples.

        Args:
            expt_df_list: expt assignment dataframe from multiple experiments.
            attributes: Columns to use for merging.

        Returns:
            result: A tuple:

                - pd.DataFrame: Merged DataFrame with experiment variant tuples.
                - dict: summary

        """
        on_cols = [UNIT_COL]
        if attributes:
            on_cols += list(attributes)

        # This will join the various experiments (outer join)
        # And generate multi-expt variants as tuples
        # It will also replace Nan generated during outer join with CATEG_NAN_VALUE
        dc_list = [DataContainer(df=df, is_df=True) for df in expt_df_list]
        expt_dc = join_expt_dfs(dc_list=dc_list, on_cols=on_cols, common_cols=[VARIANT_COL])
        expt_df = expt_dc.df

        summary = expt_df[VARIANT_COL].value_counts(normalize=True).to_dict()

        return expt_df, summary

    def apply_experiment_impact(
        self,
        df: pd.DataFrame,
        expt_metric_impacts: tuple[dict[str, float]],
        interaction_metric_impacts: dict[tuple[str], dict[str, float]] = dict(),
    ) -> pd.DataFrame:
        """
        Applies experiment impacts to metrics and includes interaction effects.

        Args:
            df: Merged DataFrame with variant tuples.
            expt_metric_impacts: Experiment impacts on metrics.
                The tuple is assumed to be ordered with k-th element which is a dict
                corresponding to eperiment k. See example.

            expt_metric_impacts = [
                {"control": {"metric1": 5}, "v1": {"metric1": -3}, "v2": {"metric1": +6}},  # Experiment 1 impact
                {"control": {"metric2": 10}, "enabled": {"metric2": 12}},  # Experiment 2 impact
            ]

            interaction_metric_impacts: Interaction impacts based on variant tuples.

        Returns:
            result: Updated DataFrame with applied impacts.
        """
        if expt_metric_impacts:
            for i, variant_impacts in enumerate(expt_metric_impacts):
                variant_col = f"{VARIANT_COL}_{i + 1}"
                for variant, impacts in variant_impacts.items():
                    for metric_name, impact_value in impacts.items():
                        df.loc[df[variant_col] == variant, metric_name] += impact_value

        if interaction_metric_impacts:
            for variant, impacts in interaction_metric_impacts.items():
                for metric_name, impact_value in impacts.items():
                    df.loc[df[VARIANT_COL] == variant, metric_name] += impact_value

        return df

    def run(self):
        """
        Runs the simulation end to end and updates the class attributes.
        """
        # Generate a population
        self.population_df, self.population_summary = self.generate_population(
            population_size=self.population_size,
            attribute_weights=self.attribute_weights,
            seed=self.population_seed,
        )

        # Sample population and run multiple experiments
        self.expt_data_list = self.simulate_expt_assignment_multi(
            population_df=self.population_df,
            expt_variant_weights_multi=self.expt_variant_weights_multi,
            population_pcnt_multi=self.population_pcnt_multi,
            non_trigger_pct_multi=self.non_trigger_pct_multi,
            seed_multi=self.expt_assignment_seed_multi,
        )

        expt_df_list = [data[0].copy() for data in self.expt_data_list]

        # Merge experiment results
        self.expt_df, self.expt_df_summary = self.merge_expt_dfs(
            expt_df_list=expt_df_list, attributes=(self.attribute_weights.keys())
        )

        # We can metrics only if metric attributes are passed.
        if self.metric_attribute_values:
            self.expt_metric_baseline_df = self.add_baseline_metrics(
                df=self.expt_df.copy(), metric_attribute_values=self.metric_attribute_values
            )

            # Apply metrics and interaction effects
            self.expt_metric_df = self.apply_experiment_impact(
                df=self.expt_metric_baseline_df.copy(),
                expt_metric_impacts=self.expt_metric_impacts,
                interaction_metric_impacts=self.interaction_metric_impacts,
            )

    def publish(self, file: Optional[TextIO] = None) -> None:
        """
        Prints information about the simulation. If an open file is passed
        then it writes the results into file.

        Args:
            file: An open file or None.
        """
        print(f"\nPopulation Summary:\n{self.population_summary}", file=file)
        print(f"\nsim.population_df:\n{self.population_df}", file=file)

        print("\nexpt_data_list (one by one):\n", file=file)
        for i, expt_data in enumerate(self.expt_data_list):
            expt_df = expt_data[0]
            summary = expt_data[1]
            print(f"\nexpt_df {i}:\n{expt_df}", file=file)
            print(f"\nsummary {i}:\n{summary}", file=file)
            print(f"\nexpt_df.shape {i}:\n{expt_df.shape}", file=file)

        print(f"\nmerged expt df (sim.expt_df):\n{self.expt_df}", file=file)
        print(f"\nsim.expt_df.shape:\n{self.expt_df.shape}", file=file)

        print(f"\nsim.expt_df_summary:\n{self.expt_df_summary}", file=file)

        if self.metric_attribute_values:
            print(
                f"\nbaseline expt metric df, sim.expt_metric_baseline_df:\n{self.expt_metric_baseline_df.head()}",
                file=file,
            )
            print(f"\nsim.expt_metric_df.shape:\n{self.expt_metric_baseline_df.shape}", file=file)

            agg_dict = {metric: "mean" for metric in self.metric_attribute_values}

            attributes = list(self.attribute_weights.keys())

            mean_attribute_metric_baseline_df = self.expt_metric_baseline_df.groupby(
                attributes
            ).agg(agg_dict)
            print(
                f"\nmean_attribute_metric_baseline_df:\n{mean_attribute_metric_baseline_df}",
                file=file,
            )

            mean_variant_metric_baseline_df = self.expt_metric_baseline_df.groupby(
                [VARIANT_COL]
            ).agg(agg_dict)
            print(f"\nmean_variant_metric_df:\n{mean_variant_metric_baseline_df}", file=file)

            print(
                f"\ndf with Impacts, sim.expt_metric_df:\n{self.expt_metric_df.head()}", file=file
            )
            print(f"\nsim.expt_metric_df.shape:\n{self.expt_metric_df.shape}", file=file)

            mean_attribute_metric_df = self.expt_metric_df.groupby(attributes).agg(agg_dict)
            print(f"\nmmean_attribute_metric_df:\n{mean_attribute_metric_df}", file=file)

            mean_variant_metric_df = self.expt_metric_df.groupby([VARIANT_COL]).agg(agg_dict)
            print(f"\nmean_variant_metric_df:\n{mean_variant_metric_df}", file=file)

            mean_attribute_variant_metric_df = self.expt_metric_df.groupby(
                attributes + [VARIANT_COL]
            ).agg(agg_dict)
            print(
                f"\nmean_attribute_variant_metric_df:\n{mean_attribute_variant_metric_df}",
                file=file,
            )
