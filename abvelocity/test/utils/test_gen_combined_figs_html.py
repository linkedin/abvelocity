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
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# Original author: Reza Hosseini

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from abvelocity.utils.gen_combined_figs_html import gen_combined_figs_html

WRITE_PATH = Path(__file__).parents[5].joinpath("docs/static/test-results/viz_utils/")
if not os.path.exists(WRITE_PATH):
    os.makedirs(WRITE_PATH)


def test_gen_combined_figs_html():
    # Create a simple DataFrame for demonstration
    data = pd.DataFrame({"Category": ["A", "B", "C", "D"], "Values": [10, 20, 30, 40]})

    # Generate a bar plot using Plotly Express
    bar_fig = px.bar(data, x="Category", y="Values", title="Test Bar Plot")

    # Generate a scatter plot
    scatter_fig = px.scatter(data, x="Category", y="Values", title="Test Scatter Plot")

    # Define hierarchical data for the Sunburst plot
    labels = ["World", "Asia", "Europe", "China", "India", "Germany", "France"]
    parents = ["", "World", "World", "Asia", "Asia", "Europe", "Europe"]
    values = [100, 60, 40, 30, 20, 25, 15]

    # Create the Sunburst plot
    sunburst_fig = go.Figure(
        go.Sunburst(labels=labels, parents=parents, values=values, branchvalues="total")
    )

    sunburst_fig.update_layout(margin=dict(t=0, l=0, r=0, b=0), title="Sunburst Plot Example")

    # Combine the plots and generate HTML
    fig_dict = {"Bar Plot": bar_fig, "Scatter Plot": scatter_fig, "Sunburst Plot": sunburst_fig}

    output_file = WRITE_PATH.joinpath("combined_figs_test.html")
    html_output = gen_combined_figs_html(fig_dict, html_file_name=output_file)

    # Assertions
    assert html_output.strip() != "", "HTML output should not be empty"
    assert output_file.exists(), "The output HTML file should exist"

    with open(output_file, "r") as f:
        file_contents = f.read()
    assert "<html>" in file_contents, "HTML file should contain <html> tag"
    assert "Test Bar Plot" in file_contents, "HTML should contain the bar plot title"
    assert "Test Scatter Plot" in file_contents, "HTML should contain the scatter plot title"
