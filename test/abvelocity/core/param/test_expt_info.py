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

import pytest
from abvelocity.core.param.expt_info import CROSS_MERGE, ExptInfo, MultiExptInfo


def test_expt_info_initialization():
    expt_info = ExptInfo(
        expt_unit_col="id",
        test_key="test123",
        experiment_id=101,
        segment_id=10,
        hash_id=1001,
        start_date="2024-01-01",
        end_date="2024-02-01",
        variants=("A", "B"),
        control_label="A",
        treatment_label="B",
        derived_stats=None,
    )

    assert expt_info.expt_unit_col == "id"
    assert expt_info.test_key == "test123"
    assert expt_info.experiment_id == 101
    assert expt_info.segment_id == 10
    assert expt_info.hash_id == 1001
    assert expt_info.start_date == "2024-01-01"
    assert expt_info.end_date == "2024-02-01"
    assert expt_info.variants == ("A", "B")
    assert expt_info.control_label == "A"
    assert expt_info.treatment_label == "B"
    assert expt_info.derived_stats is None


def test_multi_expt_info_initialization():
    expt_info1 = ExptInfo(expt_unit_col="id", test_key="test123", experiment_id=101)
    expt_info2 = ExptInfo(expt_unit_col="id", test_key="test456", experiment_id=102)
    multi_expt_info = MultiExptInfo(expt_info_list=[expt_info1, expt_info2], merge_method=CROSS_MERGE, derived_stats=None)

    assert len(multi_expt_info.expt_info_list) == 2
    assert multi_expt_info.merge_method == CROSS_MERGE
    assert multi_expt_info.derived_stats is None


def test_multi_expt_info_invalid_merge_method():
    expt_info1 = ExptInfo(expt_unit_col="id", test_key="test123", experiment_id=101)
    expt_info2 = ExptInfo(expt_unit_col="id", test_key="test456", experiment_id=102)

    with pytest.raises(NotImplementedError, match="Merge method invalid_merge is not implemented."):
        MultiExptInfo(
            expt_info_list=[expt_info1, expt_info2],
            merge_method="invalid_merge",
        )


def test_multi_expt_info_unit_col():
    expt_info1 = ExptInfo(test_key="test123", experiment_id=101)
    expt_info2 = ExptInfo(test_key="test456", experiment_id=102)
    multi_expt_info = MultiExptInfo(expt_info_list=[expt_info1, expt_info2], merge_method=CROSS_MERGE, expt_unit_col="id")

    assert multi_expt_info.expt_unit_col == "id"
    assert expt_info1.expt_unit_col == "id"
    assert expt_info2.expt_unit_col == "id"


def test_expt_info_gen_query_raises_error():
    """Test that calling gen_query on base ExptInfo class raises NotImplementedError."""
    expt_info = ExptInfo(
        expt_unit_col="id",
        test_key="test123",
        experiment_id=101,
    )

    with pytest.raises(NotImplementedError, match="`gen_query` is called in the base `ExptInfo` class"):
        expt_info.gen_query()
