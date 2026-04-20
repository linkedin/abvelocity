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

from copy import deepcopy
from typing import Dict

from abvelocity.core.journey.event.gen_event_query import EventTable, MultiEventTable
from abvelocity.core.journey.seq.seq import Seq
from abvelocity.core.journey.seq.seq_info import FULLY_DEDUPED, UNDEDUPED, SeqInfo
from abvelocity.core.param.io_param import IOParam
from abvelocity.core.param.join_query import JoinQuery


class JourneySeq(Seq):
    """Specific Seq implementation inheriting methods from the imported Seq class."""

    def gen_event_queries(self) -> Dict[str, str]:
        # Assumes gen_event_queries_via_multi_event_tables exists on the imported Seq class
        return self.gen_event_queries_via_multi_event_tables(multi_event_tables=self.multi_event_table)


JOURNEY_EVENTS = MultiEventTable(
    event_tables=[
        EventTable(table_name="ExternalData.PageView", event_label="page_view"),
        EventTable(table_name="ExternalData.PageClick", event_label="page_click"),
    ],
    common_info=EventTable(),
)
JOURNEY_SEQ_INFO_LIST = [
    SeqInfo(deduping_method=UNDEDUPED, max_seq_index=12),
    SeqInfo(deduping_method=FULLY_DEDUPED, max_seq_index=9),
]


POST_JOINS = [
    JoinQuery(
        right_table="user_table",
        join_type="LEFT",
        on=[("user", "session")],
        select_right_columns=[
            "country",
            "age",
        ],
        right_conditions=None,
        right_date_col="date",
    ),
]


# CREATE THE SINGLE GLOBAL INSTANCE CONSTANT
JOURNEY_SEQ = JourneySeq(
    create_table_prefix="rezas",
    start_date="2025-05-21-00",
    end_date="2025-05-28-00",
    # Use deepcopy to ensure the class's internal pristine copy is isolated
    seq_info_list=deepcopy(JOURNEY_SEQ_INFO_LIST),
    multi_event_table=JOURNEY_EVENTS,
    io_param=IOParam(cursor=None, print_to_html=None, save_path="", file_name_suffix=""),
    materialization=None,
    # Minimal arguments for Seq constructor
    partition_by_cols=["user"],
    conditions=[],
    post_joins=POST_JOINS,
)


def test_journey_seq_with_reset(capfd):
    """
    Tests idempotency by explicitly calling seq.reset_seq_info_list() on the
    shared JOURNEY_SEQ instance.
    """

    seq = JOURNEY_SEQ
    base_prefix = seq.create_table_prefix

    # FIRST RUN EXECUTION (Mutates seq.seq_info_list)
    seq.gen_event_queries()
    seq.gen_seq_queries()
    seq.gen_join_queries()

    # FIRST RUN ASSERTIONS
    expected_base_name = f"{base_prefix}_{UNDEDUPED}_seq"
    expected_final_name = f"{expected_base_name}_joined"

    final_table_name_run1 = seq.seq_info_list[0].output_table_name
    assert final_table_name_run1 == expected_final_name

    # SECOND RUN SETUP & EXECUTION
    seq.reset_seq_info_list()  # Fix applied here

    seq.gen_event_queries()
    seq.gen_seq_queries()
    seq.gen_join_queries()

    capfd.readouterr()

    # SECOND RUN ASSERTIONS
    final_table_name_run2 = seq.seq_info_list[0].output_table_name
    assert final_table_name_run2 == expected_final_name


def test_journey_seq_with_gen_all(capfd):
    """
    Tests idempotency using the gen_all_queries wrapper on the shared JOURNEY_SEQ instance.
    (Relies on gen_all_queries internally calling reset_seq_info_list).
    """

    seq = JOURNEY_SEQ
    base_prefix = seq.create_table_prefix
    expected_final_name = f"{base_prefix}_{UNDEDUPED}_seq_joined"

    # FIRST RUN EXECUTION
    seq.gen_all_queries()

    # FIRST RUN ASSERTIONS
    final_table_name_run1 = seq.seq_info_list[0].output_table_name
    assert final_table_name_run1 == expected_final_name

    # SECOND RUN EXECUTION
    seq.gen_all_queries()

    capfd.readouterr()

    # SECOND RUN ASSERTIONS
    final_table_name_run2 = seq.seq_info_list[0].output_table_name
    assert final_table_name_run2 == expected_final_name
