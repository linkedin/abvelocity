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

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from abvelocity.param.launch import Launch
from abvelocity.param.variant import TriggerState, Variant


@dataclass
class DerivedExptStats:
    """
    This is a dataclass to contain the derived statistics for a multi-experiment (could be univariate as well).
    """

    variants: Optional[list[Variant]] = None
    """
    A list of variants in the experiment. Each variant value could be a tuple (for multi-experiments) or a string (for simple experiments).
    """
    launches: Optional[list[Launch]] = None
    """
    A list of launches in the experiment. A launch value does not include any `CATEG_NAN_VALUE`.
    """
    non_control_launches: Optional[list[Launch]] = None
    """
    A list of "non-control" launches in the experiment.
    A non-control launch is a launch which has at least one experiment not on the `CONTROL_LABEL` arm.
    """
    trigger_states: Optional[list[TriggerState]] = None
    """
    A list of trigger states in the experiment.
    """
    variant_count_df: Optional[pd.DataFrame] = None
    """
    A dataframe containing the variant counts and percentages indexed by the variant values.
    Expected columns are:

        - constants.VARIANT_COUNT_COL
        - constants.VARIANT_PERCENT_COL
        - constants.TRIGGER_STATE_COL
        - constants.TRIGGER_STATE_COUNT_COL
        - constants.TRIGGER_STATE_PERCENT_COL
        - constants.TRIGGER_STATE_OVERALL_COL
        - constants.TRIGGER_STATE_OVERALL_COUNT_COL
        - constants.TRIGGER_STATE_OVERALL_PERCENT_COL
        - constants.VARIANT_OVER_TRIGGERED_PERCENT_COL

    """
    trigger_state_count_df: Optional[pd.DataFrame] = None
    """
    A dataframe containing the trigger state counts and percentages indexed by the trigger state values.

    Expected columns are:

            - TRIGGER_STATE_COUNT_COL
            - TRIGGER_STATE_PERCENT_COL

    """
    total_count: int = 0
    """
    Total count of units in the experiment.
    """
    total_triggered_count: int = 0
    """
    Total count of triggered units in the experiment.
    """
    total_triggered_percent: float = 0.0
    """
    Total percent of triggered units in the experiment.
    """
    conditional_trigger_dfs: Optional[dict[int, pd.DataFrame]] = None
    """
    The conditional distribution (dataframe) of the trigger states given:
        the i-th experiment (`i` is the key) is triggered.
    """
    overlap_rates: Optional[dict[int, float]] = None
    """
    The amount of overlap from other experiments on the i-th experiment
        (`i` is the key).
    """

    def __str__(self):
        variants_str = (
            ", ".join(str(variant) for variant in self.variants) if self.variants else "None"
        )
        launches_str = (
            ", ".join(str(launch) for launch in self.launches) if self.launches else "None"
        )
        non_control_launches_str = (
            ", ".join(str(launch) for launch in self.non_control_launches)
            if self.non_control_launches
            else "None"
        )
        trigger_states_str = (
            ", ".join(str(state) for state in self.trigger_states)
            if self.trigger_states
            else "None"
        )
        variant_count_str = (
            self.variant_count_df.to_string() if self.variant_count_df is not None else "None"
        )
        trigger_state_count_str = (
            self.trigger_state_count_df.to_string()
            if self.trigger_state_count_df is not None
            else "None"
        )

        conditional_trigger_dfs_str = ""
        if self.conditional_trigger_dfs:
            for i, df in self.conditional_trigger_dfs.items():
                conditional_trigger_dfs_str += f"\nexpt {i}'s overlap by other: {df.to_string()}\n"

        return (
            f"Derived Experiment Statistics:\n"
            f"- Variants: {variants_str}\n"
            f"- Launches: {launches_str}\n"
            f"- Non-Control Launches: {non_control_launches_str}\n"
            f"- Trigger States: {trigger_states_str}\n"
            f"- Variant Count DataFrame:\n{variant_count_str}\n"
            f"- Trigger State Count DataFrame:\n{trigger_state_count_str}\n"
            f"- Total Count: {self.total_count}\n"
            f"- Total Triggered Count: {self.total_triggered_count}\n"
            f"- Total Triggered Percent: {self.total_triggered_percent}\n"
            f"- Conditional Trigger Dfs: {conditional_trigger_dfs_str} \n"
            f"- overlap Rates: {self.overlap_rates} \n"
        )

    def to_html(self):
        variants_str = (
            ", ".join(str(variant) for variant in self.variants) if self.variants else "None"
        )
        launches_str = (
            ", ".join(str(launch) for launch in self.launches) if self.launches else "None"
        )
        non_control_launches_str = (
            ", ".join(str(launch) for launch in self.non_control_launches)
            if self.non_control_launches
            else "None"
        )
        trigger_states_str = (
            ", ".join(str(state) for state in self.trigger_states)
            if self.trigger_states
            else "None"
        )
        variant_count_html = (
            self.variant_count_df.to_html() if self.variant_count_df is not None else "<p>None</p>"
        )
        trigger_state_count_html = (
            self.trigger_state_count_df.to_html()
            if self.trigger_state_count_df is not None
            else "<p>None</p>"
        )

        conditional_trigger_dfs_html = ""
        if self.conditional_trigger_dfs:
            for i, df in self.conditional_trigger_dfs.items():
                conditional_trigger_dfs_html += (
                    f"<p>expt {i}'s overlap by other:</p> <p>{df.to_html()}</p>"
                )

        overlap_rates_html = ""
        if self.overlap_rates:
            overlap_rates_html = (
                "<table>\n"
                "  <thead>\n"
                "    <tr><th>Expt</th><th>Overlap Rate (%)</th></tr>\n"
                "  </thead>\n"
                "  <tbody>\n"
                + "\n".join(
                    f"    <tr><td>Expt {key}'s overlap: </td><td>{round(value, 2)}%</td></tr>"
                    for key, value in self.overlap_rates.items()
                )
                + "\n  </tbody>\n</table>"
            )

        return (
            f"<div>"
            f"<h3>Derived Experiment Statistics</h3>"
            f"<p><strong>Variants:</strong> {variants_str}</p>"
            f"<p><strong>Launches:</strong> {launches_str}</p>"
            f"<p><strong>Non-Control Launches:</strong> {non_control_launches_str}</p>"
            f"<p><strong>Trigger States:</strong> {trigger_states_str}</p>"
            f"<h4>Variant Count DataFrame:</h4>{variant_count_html}"
            f"<h4>Trigger State Count DataFrame:</h4>{trigger_state_count_html}"
            f"<p><strong>Total Count:</strong> {self.total_count}</p>"
            f"<p><strong>Total Triggered Count:</strong> {self.total_triggered_count}</p>"
            f"<p><strong>Total Triggered Percent:</strong> {self.total_triggered_percent}</p>"
            f"<p><strong>Conditional Trigger Dfs:</strong></p>"
            f"<p>{conditional_trigger_dfs_html}</p>"
            f"<p><strong>Overlap Rates:</strong><p>"
            f"<p>{overlap_rates_html}</p>"
            f"</div>"
        )

    def to_markdown(self) -> str:
        """
        Convert the DerivedExptStats data to a Markdown format.
        """
        variants_str = (
            ", ".join(str(variant) for variant in self.variants) if self.variants else "None"
        )
        launches_str = (
            ", ".join(str(launch) for launch in self.launches) if self.launches else "None"
        )
        non_control_launches_str = (
            ", ".join(str(launch) for launch in self.non_control_launches)
            if self.non_control_launches
            else "None"
        )
        trigger_states_str = (
            ", ".join(str(state) for state in self.trigger_states)
            if self.trigger_states
            else "None"
        )
        variant_count_md = (
            self.variant_count_df.to_markdown() if self.variant_count_df is not None else "None"
        )
        trigger_state_count_md = (
            self.trigger_state_count_df.to_markdown()
            if self.trigger_state_count_df is not None
            else "None"
        )

        conditional_trigger_dfs_md = ""
        if self.conditional_trigger_dfs:
            for i, df in self.conditional_trigger_dfs.items():
                conditional_trigger_dfs_md += f"\nexpt {i}'s overlap by other: {df.to_markdown()}\n"

        overlap_rates_md = ""
        if self.overlap_rates:
            overlap_rates_md = "| Expt | Overlap Rate (%) |\n|---|---|\n" + "\n".join(
                f"| Expt {key}'s overlap: | {round(value, 2)}% |"
                for key, value in self.overlap_rates.items()
            )

        markdown_str = (
            f"### Derived Experiment Statistics\n"
            f"- **Variants**: {variants_str}\n"
            f"- **Launches**: {launches_str}\n"
            f"- **Non-Control Launches**: {non_control_launches_str}\n"
            f"- **Trigger States**: {trigger_states_str}\n"
            f"#### Variant Count DataFrame:\n{variant_count_md}\n"
            f"#### Trigger State Count DataFrame:\n{trigger_state_count_md}\n"
            f"- **Total Count**: {self.total_count}\n"
            f"- **Total Triggered Count**: {self.total_triggered_count}\n"
            f"- **Total Triggered Percent**: {self.total_triggered_percent}\n"
            f"- **Conditional Trigger Dfs:\n{self.conditional_trigger_dfs}\n"
            f"- **Overlap Rates:\n {overlap_rates_md}\n"
        )

        return markdown_str
