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

"""Table materialization configuration primitives.

Defines the supported materialization strategies and the dataclass used to
configure how a `TableQuery` should be executed (full overwrite, incremental
partition overwrite, or as a temporary view).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from mashumaro.mixins.json import DataClassJSONMixin


class MaterializationType(str, Enum):
    """Supported table materialization strategies."""

    OVERWRITE = "overwrite"  # Full rebuild (DROP + CREATE TABLE AS ...)
    INCREMENTAL = "incremental"  # Insert overwrite a specific partition
    TEMP_VIEW = "temp_view"  # Create or replace a (temporary) view


class QueryEngine(str, Enum):
    """Supported query execution engines for materialization logic."""

    TRINO = "trino"
    SPARK = "spark"
    CUSTOM_SPARK = "custom_spark"  # Blah-specific Spark with Presto->Spark SQL conversion


@dataclass
class TableMaterialization(DataClassJSONMixin):
    """Configuration governing how `TableQuery` SQL is materialized.

    Attributes:
        materialization_type: One of MaterializationType values.
        partition_col: Required for INCREMENTAL (partition column name).
        partition_value: Required for INCREMENTAL (the partition value to overwrite).
        engine: Which query engine the generated SQL will target (affects dialect nuances).
    """

    materialization_type: MaterializationType = MaterializationType.OVERWRITE
    partition_col: Optional[str] = None
    partition_value: Optional[str] = None
    engine: QueryEngine = QueryEngine.TRINO
    session_variables_applied: bool = field(default=False, init=False, repr=False)

    def __init__(
        self,
        materialization_type: MaterializationType = MaterializationType.OVERWRITE,
        partition_col: Optional[str] = None,
        partition_value: Optional[str] = None,
        engine: QueryEngine = QueryEngine.TRINO,
        io_param: Optional[Any] = None,
    ) -> None:
        # Assign configuration using direct attribute assignment
        self.materialization_type = materialization_type
        self.partition_col = partition_col
        self.partition_value = partition_value
        self.engine = engine
        self.session_variables_applied = False
        # Optionally apply engine session (one-time) without storing io_param
        if io_param is not None:
            self.apply_engine_session(io_param)

    def validate(self) -> "TableMaterialization":
        if self.materialization_type == MaterializationType.INCREMENTAL:
            if not self.partition_col or self.partition_value is None:
                raise ValueError("Incremental materialization requires partition_col and partition_value")
        return self

    def apply_engine_session(self, io_param: Any) -> None:
        """One-time engine session setup.

        Currently only needed for Trino incremental to enable overwrite semantics for existing partitions.
        Safe to call multiple times; executes at most once per instance.
        """
        if (
            not self.session_variables_applied
            and self.materialization_type == MaterializationType.INCREMENTAL
            and self.engine == QueryEngine.TRINO
            and io_param is not None
        ):
            io_param.cursor.execute_and_fetchall("""SET SESSION hive.insert_existing_partitions_behavior = 'OVERWRITE'""")
            print("Session variables applied for Trino incremental insert overwrite.")
            self.session_variables_applied = True
