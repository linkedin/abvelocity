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

"""Shared mashumaro serialization utilities.

Provides a reusable :class:`DataFrameSerializationStrategy` for round-tripping
``pd.DataFrame`` fields through JSON, and a :class:`DataFrameConfig` base config
that registers it. Dataclasses with ``pd.DataFrame`` fields should inherit from
``DataClassJSONMixin`` and declare ``class Config(DataFrameConfig): pass``.

Example::

    from dataclasses import dataclass
    from typing import Optional

    import pandas as pd
    from mashumaro.mixins.json import DataClassJSONMixin

    from abvelocity.core.utils.serialization import DataFrameConfig


    @dataclass
    class MyResult(DataClassJSONMixin):
        df: Optional[pd.DataFrame] = None

        class Config(DataFrameConfig):
            pass
"""

import pandas as pd
from mashumaro.config import BaseConfig
from mashumaro.types import SerializationStrategy


class DataFrameSerializationStrategy(SerializationStrategy):
    """Serializes a ``pd.DataFrame`` to/from a JSON-safe dict using ``orient="split"``.

    The ``split`` format preserves column names, row index, and data faithfully,
    making it the most reliable choice for round-tripping DataFrames through JSON.
    """

    def serialize(self, value: pd.DataFrame) -> dict:
        return value.to_dict(orient="split")

    def deserialize(self, value: dict) -> pd.DataFrame:
        return pd.DataFrame(
            data=value["data"],
            columns=value["columns"],
            index=value["index"],
        )


class DataFrameConfig(BaseConfig):
    """mashumaro ``BaseConfig`` subclass that registers :class:`DataFrameSerializationStrategy`.

    Handles all ``pd.DataFrame`` fields, including ``Optional[pd.DataFrame]``
    and container types such as ``dict[str, pd.DataFrame]`` or ``list[pd.DataFrame]``.
    """

    serialization_strategy = {pd.DataFrame: DataFrameSerializationStrategy()}
