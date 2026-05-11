# BSD 2-CLAUSE LICENSE
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
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
"""Abstract base class for post-fetch DataFrame transformations.

Lives in its own module so :class:`TSMetricsConfig` can annotate
``post_fetch_transforms: Sequence[TSTransform]`` without importing the
concrete-transforms module — and so the concrete-transforms module can
freely import ``TSMetricsConfig`` for its method signatures.  The
dependency graph stays a DAG:

    ts_transform.py  ←  ts_metrics_config.py
            ↑                    ↑
            └─── transforms.py ──┘

The ``apply`` method's ``ts_config`` and ``metric_info`` parameters are
intentionally NOT type-annotated here — adding hints would force this
module to import ``TSMetricsConfig`` and re-introduce a cycle.  Concrete
subclasses (in ``transforms.py``) can annotate freely.
"""

from abc import ABC, abstractmethod

import pandas as pd


class TSTransform(ABC):
    """Post-fetch transformation applied to a wide-format metrics DataFrame.

    Transforms compose into an ordered list on
    :class:`~abvelocity.ts.get_data.ts_metrics_config.TSMetricsConfig`
    and run inside ``TSMetricsQuery.get_df`` after the ``DimCollapser``
    stage.

    Implementations must NOT mutate the input frame in place — return a
    new DataFrame.
    """

    @abstractmethod
    def apply(self, df: pd.DataFrame, ts_config, metric_info) -> pd.DataFrame:
        """Return a transformed DataFrame.

        Args:
            df: Wide-format frame with one row per ``(time_alias × dims)``.
            ts_config: :class:`TSMetricsConfig` for the source query.
                Provides ``time_alias`` and the source ``freq``.
            metric_info: :class:`MetricInfo` for the source query.
                Provides the metric value columns and the ``dims`` list.
        """

    def str_name(self) -> str:
        """Short text identifier for this transform — used to auto-derive
        output ``metric_id`` / ``metric_name`` strings on
        :class:`~abvelocity.ts.li.forecast_jobs.JobConfig`.

        Default is the lowercased class name; concrete subclasses
        override for descriptive forms like ``"weekly"`` or
        ``"wow_diff"``.
        """
        return type(self).__name__.lower()
