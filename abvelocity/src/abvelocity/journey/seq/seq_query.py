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

from abvelocity.journey.seq.add_seq_elements_as_cols import add_seq_elements_as_cols
from abvelocity.journey.seq.gen_deduped_seq_query import gen_deduped_seq_query
from abvelocity.journey.seq.gen_event_count_map_query import gen_event_count_map_query
from abvelocity.journey.seq.gen_fully_deduped_seq_query import gen_fully_deduped_seq_query
from abvelocity.journey.seq.gen_undeduped_seq_query import gen_undeduped_seq_query
from abvelocity.journey.seq.seq_info import (
    CONSECUTIVE_DEDUPED,
    DEDUPING_MTHODS,
    FULLY_DEDUPED,
    MAP,
    UNDEDUPED,
)


@dataclass
class SeqQuery:
    """
    A class to generate SQL queries for sequencing events.
    """

    event_table_name: str
    # SQL table with event data

    time_col: str
    # Time column used to decide the order of events in the sequence

    event_col: str
    # Event column used as the elements of the sequence

    partition_by_cols: list[str]
    # Columns to partition by before sequencing the data

    deduping_method: str
    # Either 'consecutive' or 'fully' for deduplication

    output_table_name: Optional[str] = None
    # Name of the output sequence table at the unit level

    max_seq_index: Optional[int] = None
    # Maximum sequence index, if applicable

    order_list: Optional[list[str]] = None
    """
    A list specifying the desired order of events in the output array.
    If provided, the query will order the events in the array according to this list.
    If None, the events will be ordered by their timestamp.
    """

    query: Optional[str] = None
    """The query can be directly provided here. Also each time `self.gen` is called the result will be stored here."""

    def gen(self) -> str:
        """
        Generate the appropriate query based on the deduping method.

        Returns:
            str: The generated SQL query.
        """
        if self.deduping_method == CONSECUTIVE_DEDUPED:
            self.query = gen_deduped_seq_query(
                table_name=self.event_table_name,
                partition_by_cols=self.partition_by_cols,
                time_col=self.time_col,
                event_col=self.event_col,
            )
        elif self.deduping_method == FULLY_DEDUPED:
            self.query = gen_fully_deduped_seq_query(
                table_name=self.event_table_name,
                partition_by_cols=self.partition_by_cols,
                time_col=self.time_col,
                event_col=self.event_col,
                order_list=self.order_list,
            )
        elif self.deduping_method == UNDEDUPED:
            self.query = gen_undeduped_seq_query(
                table_name=self.event_table_name,
                partition_by_cols=self.partition_by_cols,
                time_col=self.time_col,
                event_col=self.event_col,
                order_list=self.order_list,
            )
        elif self.deduping_method == MAP:
            self.query = gen_event_count_map_query(
                table_name=self.event_table_name,
                partition_by_cols=self.partition_by_cols,
                time_col=self.time_col,
                event_col=self.event_col,
            )
        else:
            raise ValueError(f"deduping_method must be in {DEDUPING_MTHODS}")

        if self.max_seq_index is not None and self.deduping_method != "map":
            self.query = add_seq_elements_as_cols(
                base_query=self.query,
                partition_by_cols=self.partition_by_cols,
                max_seq_index=self.max_seq_index,
            )

        if self.output_table_name is not None:
            self.query = f"""
                DROP TABLE IF EXISTS {self.output_table_name};
                CREATE TABLE IF NOT EXISTS {self.output_table_name} AS
                {self.query}
            """

        return self.query
