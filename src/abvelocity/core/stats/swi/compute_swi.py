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

from typing import Optional

import pandas as pd
from abvelocity.core.param.constants import METRIC_NAME_COL
from abvelocity.core.stats.swi.swi_ratio import swi_ratio
from abvelocity.core.stats.swi.swi_simple import swi_simple

SITE_WIDE_IMPACT_COL = "site_wide_impact"
SITE_WIDE_IMPACT_PERC_COL = "site_wide_impact_perc"

# Columns expected in variant_effect_df (to be populated in a later PR).
CONTROL_METRIC_COL = "control_metric_total"
TREATMENT_METRIC_COL = "treatment_metric"
IMPACTED_POP_COUNT_COL = "impacted_pop_count"
DELTA_COL = "delta"

# Columns expected in complement_metrics_df.
COMPLEMENT_NUMER_COL = "numer"
COMPLEMENT_DENOM_COL = "denom"


def compute_swi(
    variant_effect_df: pd.DataFrame,
    complement_metrics_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Computes site-wide impact (SWI) for each comparison in variant_effect_df.

    SWI answers: what is the lift in the average metric across all members in the
    population if we apply treatment to all triggered/targeted members (ramp to 100%)?
    This makes experiment results comparable across experiments with different
    triggering or targeting rates.

    **Inputs from variant_effect_df:**

    - ``metric``: metric name.
    - ``delta``: per-unit treatment effect; scalar for simple metrics,
      tuple ``(delta_numer, delta_denom)`` for ratio metrics.
    - ``control_metric_total``: total aggregated metric for the **control arm** of the
      triggered/targeted segment; scalar for simple, tuple ``(numer, denom)``
      for ratio.
    - ``treatment_metric``: total aggregated metric for the treatment arm (carried
      for reference; not required by the SWI formula itself).
    - ``impacted_pop_count``: total triggered/targeted population
      (treatment + control arm combined).

    **complement_metrics_df schema:**

    Same long-format style as ``variant_effect_df`` — one row per metric with a
    fixed set of columns:

    +--------+----------+----------+
    | metric | numer    | denom    |
    +--------+----------+----------+
    | clicks | 30000.0  | (absent) |
    | ses... | 6000.0   | 3000.0   |
    +--------+----------+----------+

    Duplicate metric names raise ``ValueError``.  When ``complement_metrics_df`` is
    ``None`` or a metric is absent from it, ``control_metric_total`` is used as the sole
    baseline for ``site_wide_impact_perc`` (best effort).

    Uses the potential outcomes framework.  ``*_pop_control`` quantities are the
    population-level metric in the counterfactual control world (all triggered units
    stay on control).  ``*_pop_treatment`` quantities are the population-level metric
    in the treatment world (triggered units receive treatment, complement is unchanged).

    **SWI formulas — simple metrics (scalars):**

        swi_raw           = delta * impacted_pop_count
        swi_relative_perc = swi_raw / numer_pop_total_given_control * 100

    where ``numer_pop_total_given_control = control_metric_total + complement_metric_total``
    (complement absent: ``numer_pop_total_given_control = control_metric_total``).

    **SWI formulas — ratio metrics (tuples):**

        numer_pop_total_given_control = control_numer_total + complement_numer_total
        denom_pop_total_given_control = control_denom_total + complement_denom_total
        numer_pop_total_given_treat   = numer_pop_total_given_control + delta_numer * impacted_pop_count
        denom_pop_total_given_treat   = denom_pop_total_given_control + delta_denom * impacted_pop_count
        swi_raw           = numer_pop_total_given_treat / denom_pop_total_given_treat - numer_pop_total_given_control / denom_pop_total_given_control
        swi_relative_perc = swi_raw / (numer_pop_total_given_control / denom_pop_total_given_control) * 100

    Ratio detection: ``delta`` being a tuple marks the metric as a ratio metric.

    Args:
        variant_effect_df: Combined MEA effect DataFrame. Required columns:
            ``metric``, ``delta``, ``impacted_pop_count``, ``control_metric_total``.
        complement_metrics_df: One row per metric with columns ``metric``, ``numer``,
            and optionally ``denom``. Duplicate metric names raise ``ValueError``.
            When ``None``, ``control_metric_total`` alone is used as the baseline.

    Returns:
        Copy of ``variant_effect_df`` with two new columns:
        ``site_wide_impact`` (absolute) and ``site_wide_impact_perc`` (relative %).
        Rows with missing ``delta`` or ``impacted_pop_count`` receive NaN.
    """
    if complement_metrics_df is not None:
        if complement_metrics_df[METRIC_NAME_COL].duplicated().any():
            dupes = complement_metrics_df[METRIC_NAME_COL][complement_metrics_df[METRIC_NAME_COL].duplicated()].tolist()
            raise ValueError(f"complement_metrics_df has duplicate metric names: {dupes}. " f"Each metric must appear in exactly one row.")
        complement_metrics_lookup_df = complement_metrics_df.set_index(METRIC_NAME_COL)
    else:
        complement_metrics_lookup_df = None

    df = variant_effect_df.copy()
    df[SITE_WIDE_IMPACT_COL] = float("nan")
    df[SITE_WIDE_IMPACT_PERC_COL] = float("nan")

    for idx, row in df.iterrows():
        metric_name = row[METRIC_NAME_COL]
        delta = row.get(DELTA_COL)
        impacted_pop_count = row.get(IMPACTED_POP_COUNT_COL)
        control_metric_total = row.get(CONTROL_METRIC_COL)

        if delta is None or impacted_pop_count is None or pd.isna(impacted_pop_count):
            print(f"\n*** compute_swi: skipping metric '{metric_name}' — " "delta or impacted_pop_count is missing.")
            continue

        complement_row = (
            complement_metrics_lookup_df.loc[metric_name]
            if complement_metrics_lookup_df is not None and metric_name in complement_metrics_lookup_df.index
            else None
        )

        if isinstance(delta, tuple):
            complement_metric_total = None
            if complement_row is not None:
                cn = complement_row.get("numer")
                cd = complement_row.get("denom")
                complement_metric_total = (
                    cn if cn is not None and not pd.isna(cn) else 0.0,
                    cd if cd is not None and not pd.isna(cd) else 0.0,
                )
            swi_raw, swi_perc = swi_ratio(
                delta=delta,
                impacted_pop_count=impacted_pop_count,
                control_metric_total=control_metric_total,
                complement_metric_total=complement_metric_total,
            )
        else:
            complement_metric_total = None
            if complement_row is not None:
                cn = complement_row.get("numer")
                if cn is not None and not pd.isna(cn):
                    complement_metric_total = cn
            swi_raw, swi_perc = swi_simple(
                delta=delta,
                impacted_pop_count=impacted_pop_count,
                control_metric_total=control_metric_total,
                complement_metric_total=complement_metric_total,
            )

        df.at[idx, SITE_WIDE_IMPACT_COL] = swi_raw
        df.at[idx, SITE_WIDE_IMPACT_PERC_COL] = swi_perc

    return df
