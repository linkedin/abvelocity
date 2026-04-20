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

import pandas as pd
import pytest
from abvelocity.core.utils.serialization import DataFrameSerializationStrategy


@pytest.fixture
def sample_df():
    return pd.DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]}, index=[10, 20, 30])


def test_dataframe_strategy_serialize(sample_df):
    strategy = DataFrameSerializationStrategy()
    result = strategy.serialize(sample_df)

    assert isinstance(result, dict)
    assert set(result.keys()) == {"columns", "data", "index"}
    assert result["columns"] == ["a", "b"]
    assert result["index"] == [10, 20, 30]


def test_dataframe_strategy_deserialize(sample_df):
    strategy = DataFrameSerializationStrategy()
    serialized = strategy.serialize(sample_df)
    result = strategy.deserialize(serialized)

    pd.testing.assert_frame_equal(result, sample_df)


def test_dataframe_strategy_round_trip_preserves_index(sample_df):
    strategy = DataFrameSerializationStrategy()
    result = strategy.deserialize(strategy.serialize(sample_df))

    assert list(result.index) == [10, 20, 30]
    assert list(result.columns) == ["a", "b"]


def test_dataframe_strategy_round_trip_tuple_column():
    # The strategy round-trips through a Python dict, so tuples are preserved as-is.
    # Note: in a full to_json()/from_json() round-trip, tuples become lists (JSON has no tuple type).
    df = pd.DataFrame({"variant": [("control",), ("enabled", "v2")]})
    strategy = DataFrameSerializationStrategy()
    result = strategy.deserialize(strategy.serialize(df))

    assert result["variant"].tolist() == [("control",), ("enabled", "v2")]
