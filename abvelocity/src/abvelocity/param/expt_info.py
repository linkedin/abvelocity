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

from abvelocity.param.derived_expt_stats import DerivedExptStats

UNION_MERGE = "union"
"""This merging method will concatenate the dataframes of the experiments vertically.
This is useful if an experiment is run on different segments and the user wants to analyze them together.
This will simply take the union of the assignment data vertically.
TODO: This is not implemented yet.
"""
CROSS_MERGE = "cross"
"""This merging method will create a cross variant for each unit of experiment.
For example if

    - experiment 1 has variants A and B
    - experiment 2 has variants C and D

Then the cross merge will create 9 variants because we do allow for "nan"s in the assignment data (to denote no trigger for that experiment).
The set of all possible cross-variants are:

    - (A, C)
    - (A, D)
    - (A, nan)
    - (B, C)
    - (B, D)
    - (B, nan)
    - (nan, C)
    - (nan, D)
    - (nan, nan)

"""

IMPLEMENTED_MERGE_METHODS = (CROSS_MERGE,)


@dataclass
class ExptInfo:
    """
    This is a dataclass to contain the information for one experiment in the analysis.
    """

    name: Optional[str] = None
    """
    name of the experiment. This is used merely in reports etc.
    If None, test_key will be used in it's place.
    """
    test_key: Optional[str] = None
    """
    Test key of the experiment.
    """
    experiment_id: Optional[int] = None
    """
    Experiment (iteration) ID.
    """
    segment_id: Optional[int] = None
    """
    Segment ID.
    """
    hash_id: Optional[int] = None
    """
    hash ID for the experiment (hashes units to variants).
    This field should be optional in most use cases.
    """
    start_date: Optional[str] = None
    """
    Start Date in this format: YYYY-MM-dd-00"""
    end_date: Optional[str] = None
    """
    End date in this format: YYYY-MM-dd-00.
    """
    variant_map: Optional[dict[str, str]] = None
    """
    A dictionary to map a variant name to a new name.
    This is useful to combine various variants into one,
    or improving on existing naming in data.
    As an example an unusual control label can be converetd here.
    """
    variants: Optional[tuple[str]] = None
    """
    The variants that the user is interested in.
    This is an optional field and can be utilized
    to only choose needed variants.
    """
    control_label: Optional[str] = "control"
    """
    The key for the control arm (variant).
    That key will be mapped to "control" in the SQL query.
    """
    treatment_label: Optional[str] = "enabled"
    """
    The key for the treatment arm (variant).
    That key will be mapped to "enabled" in the SQL query.
    # TODO: This field is not used in the current implementation.
    """
    query: Optional[str] = None
    """
    The query to get experiment assignment data.
    If this field is specified, then this will be used to get the data.
    This is especially useful to specify more complex queries to get assignmenmt data.
    For example if one experiment data is stored across test keys / segments etc. That might require a complex join and this field allows users to
    pass their own query.
    """
    derived_stats: Optional[DerivedExptStats] = None
    """
    Derived statistics for the experiment.
    This field can only be populated after the expt assignment data is extracted.
    """

    def __str__(self):
        variants_str = ", ".join(self.variants) if self.variants else "None"
        derived_stats_str = str(self.derived_stats) if self.derived_stats else "None"
        return (
            f"Experiment Information:\n"
            f"- Test Key: {self.test_key}\n"
            f"- Experiment ID: {self.experiment_id}\n"
            f"- Segment ID: {self.segment_id}\n"
            f"- Hash ID: {self.hash_id}\n"
            f"- Start Date: {self.start_date}\n"
            f"- End Date: {self.end_date}\n"
            f"- Variants Mapping: {self.variant_map}\n"
            f"- Variants: {variants_str}\n"
            f"- Control Label: {self.control_label}\n"
            f"- Treatment Label: {self.treatment_label}\n"
            f"- query: {self.query}\n"
            f"- Derived Stats:\n{derived_stats_str}\n"
        )

    def to_html(self):
        variants_str = ", ".join(self.variants) if self.variants else "None"
        derived_stats_html = self.derived_stats.to_html() if self.derived_stats else "<p>None</p>"
        return (
            f"<div>"
            f"<h3>Experiment Information</h3>"
            f"<p><strong>Test Key:</strong> {self.test_key}</p>"
            f"<p><strong>Experiment ID:</strong> {self.experiment_id}</p>"
            f"<p><strong>Segment ID:</strong> {self.segment_id}</p>"
            f"<p><strong>Hash ID:</strong> {self.hash_id}</p>"
            f"<p><strong>Start Date:</strong> {self.start_date}</p>"
            f"<p><strong>End Date:</strong> {self.end_date}</p>"
            f"<p><strong>Variants Mapping:</strong> {self.variant_map}</p>"
            f"<p><strong>Variants:</strong> {variants_str}</p>"
            f"<p><strong>Control Label:</strong> {self.control_label}</p>"
            f"<p><strong>Treatment Label:</strong> {self.treatment_label}</p>"
            f"<p><strong>Query:</strong> {self.query}</p>"
            f"<h4>Derived Stats:</h4>{derived_stats_html}"
            f"</div>"
        )

    def to_markdown(self) -> str:
        """
        Convert the dataclass fields and values to a Markdown format with headings and values.
        """
        variants_str = ", ".join(self.variants) if self.variants else "None"
        derived_stats_md = self.derived_stats.to_markdown() if self.derived_stats else "None"

        markdown_str = "### Experiment Information\n"
        markdown_str += f"- **Test Key**: {self.test_key}\n"
        markdown_str += f"- **Experiment ID**: {self.experiment_id}\n"
        markdown_str += f"- **Segment ID**: {self.segment_id}\n"
        markdown_str += f"- **Hash ID**: {self.hash_id}\n"
        markdown_str += f"- **Start Date**: {self.start_date}\n"
        markdown_str += f"- **End Date**: {self.end_date}\n"
        markdown_str += f"- **Variants Mapping**: {self.variant_map}\n"
        markdown_str += f"- **Variants**: {variants_str}\n"
        markdown_str += f"- **Control Label**: {self.control_label}\n"
        markdown_str += f"- **Treatment Label**: {self.treatment_label}\n"
        markdown_str += f"- **Query**: {self.query}\n"
        markdown_str += f"### Derived Stats\n{derived_stats_md}\n"

        return markdown_str

    def __post_init__(self):
        if self.name is None:
            if self.test_key is not None:
                self.name = self.test_key
            else:
                self.name = "Expt"


@dataclass
class MultiExptInfo:
    expt_info_list: list[ExptInfo]
    """A list of single experiments info."""
    merge_method: Optional[str] = None
    """The method to merge the individual experiments data.
    The only supported method is `CROSS_MERGE = "cross"` which will create tuples
    for each unit to represent the exposed variants (or no trigger) for each of
    experiments given in `expt_info_list`.
    """
    derived_stats: Optional[DerivedExptStats] = None
    """
    The derived statistics for the multi-experiment.
    """

    def __post_init__(self):
        """Checks to see if merge method is implemented."""
        if self.merge_method is not None and self.merge_method not in IMPLEMENTED_MERGE_METHODS:
            raise NotImplementedError(f"Merge method {self.merge_method} is not implemented.")

    def __str__(self):
        expt_info_str = "\n\n".join(str(expt) for expt in self.expt_info_list)
        derived_stats_str = str(self.derived_stats) if self.derived_stats else "None"
        return (
            f"Multiple Experiment Information:\n"
            f"{expt_info_str}\n"
            f"- Merge Method: {self.merge_method}\n"
            f"- Derived Stats:\n{derived_stats_str}"
        )

    def to_html(self):
        """
        Convert the MultiExptInfo data to a html format.
        """
        expt_info_html = "".join(expt.to_html() for expt in self.expt_info_list)
        derived_stats_html = self.derived_stats.to_html() if self.derived_stats else "<p>None</p>"
        return (
            f"<div>"
            f"<h3>Multiple Experiment Information</h3>"
            f"{expt_info_html}"
            f"<p><strong>Merge Method:</strong> {self.merge_method}</p>"
            f"<h4>Derived Stats:</h4>{derived_stats_html}"
            f"</div>"
        )

    def to_markdown(self) -> str:
        """
        Convert the MultiExptInfo data to a Markdown format.
        """
        expt_info_md = "\n".join(expt.to_markdown() for expt in self.expt_info_list)
        derived_stats_md = self.derived_stats.to_markdown() if self.derived_stats else "None"

        markdown_str = (
            f"### Multiple Experiment Information\n"
            f"{expt_info_md}\n"
            f"- **Merge Method**: {self.merge_method}\n"
            f"### Derived Stats\n{derived_stats_md}\n"
        )

        return markdown_str
