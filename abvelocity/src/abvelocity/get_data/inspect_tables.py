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


import os
from typing import List

from abvelocity.param.io_param import IOParam
from abvelocity.testing.desc_table import desc_table


def inspect_tables(
    table_names: List[str],
    io_param: IOParam,
    count_distinct_cols: List[str],
    distinct_values_cols: List[str],
):
    """
    Inspects tables using desc_table and prints the results.

    Args:
        table_names: A list of table names to inspect.
        io_param: An IOParam instance containing the cursor and save_path.
        count_distinct_cols: List of columns to count distinct values for.
        distinct_values_cols: List of columns to retrieve distinct values for.
    """
    for table_name in table_names:
        print(f"\n\n\n*** describe: {table_name}")
        log_file = os.path.join(
            io_param.save_path, (table_name).replace(".", "_")[10:]
        )  # use os.path.join
        desc_dict = desc_table(
            cursor=io_param.cursor,
            table_name=table_name,
            count_distinct_cols=count_distinct_cols,
            distinct_values_cols=distinct_values_cols,
            log_file=str(log_file),  # cast to string
        )

        for k, v in desc_dict.items():
            print(f"\n*** k: {k}")
            print(v)
