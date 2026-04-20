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
# author: Reza Hosseini
from typing import Optional

import pandas as pd


def df_to_markdown(
    df: pd.DataFrame,
    heading: Optional[str] = None,
    top_paragraphs: list[str] = [],
    file_name: Optional[str] = None,
    caption: str = "",
) -> str:
    """Converts a pandas dataframe to a markdown string.
        This is useful to generate quick readable markdown reports.

    Args:
        df: A pandas dataframe.
        heading: A string for the heading of the markdown.
            The default is None, in which case no heading will be included.
        top_paragraphs: A list of strings for the top paragraphs of the markdown.
            The default is an empty list.
        file_name: A string for the name of the markdown file.
            If None, no file will be written.
        caption: A string for the caption of the table.
            The default is an empty string.

    Returns:
        A string of the markdown.
    """
    # Create the markdown table representation
    df_md = df.to_markdown(index=False)

    # Adding the caption if provided
    if caption:
        df_md = f"**{caption}**\n\n" + df_md

    md_str = ""
    if heading is not None:
        md_str += f"# {heading}\n\n"

    for paragraph in top_paragraphs:
        md_str += f"{paragraph}\n\n"

    md_str += df_md + "\n"

    if file_name is not None:
        with open(file_name, "w") as f:
            f.write(md_str)
            print(f"{file_name} was created.")

    return md_str
