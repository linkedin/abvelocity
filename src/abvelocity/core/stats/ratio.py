# BSD 2-CLAUSE LICENSE
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# Original author: Reza Hosseini

from typing import Dict

import numpy as np
import pandas as pd
from abvelocity.core.stats.normal_ci import calc_standard_normal_ci


def calc_ratio_stats(df: pd.DataFrame, numer_col: str, denom_col: str, ci_coverage: float = 0.95) -> Dict[str, float]:
    """
    Computes the ratio estimate (mean of numer_col / mean of denom_col), its
    approximate variance using the delta method, and a confidence interval using
    the normal approximation (see e.g., Oehlert, G. W. 1992. A note on the delta
    method. American Statistician 46: 27–29).

    We assume each unit of the experiment is given in a separate row and they are independent.

    Args:
        df (pd.DataFrame): The full dataset.
        numer_col (str): The column name for the numerator variable.
        denom_col (str): The column name for the denominator variable.
        ci_coverage (float): The desired coverage of the confidence interval, a value
            between 0 and 1. Defaults to 0.95.

    Returns:
        Dict[str, float]: A dictionary containing:
            - "estimator_value": The point estimate of the ratio.
            - "estimator_var": The approximate variance using the delta method.
            - "ci": A tuple of two floats, representing the lower and upper bounds of
              the confidence interval (normal approximation).

    Raises:
        ValueError: If specified columns are not found in the DataFrame.
        ValueError: If DataFrame has fewer than 2 rows.
        ValueError: If mean of denominator column is zero or near zero.
        ValueError: If ci_coverage is not between 0 and 1.
    """
    if not all(col in df.columns for col in [numer_col, denom_col]):
        raise ValueError("Specified columns not found in the DataFrame.")

    n = len(df)
    if n < 2:
        raise ValueError("DataFrame must have at least 2 rows for variance estimation.")

    mean_numer = df[numer_col].mean()
    mean_denom = df[denom_col].mean()

    if np.isclose(mean_denom, 0):
        raise ValueError("Mean of denominator column is zero or near zero.")

    ratio_estimate = mean_numer / mean_denom

    # Unbiased sample variances and covariance
    var_numer = df[numer_col].var(ddof=1)
    var_denom = df[denom_col].var(ddof=1)
    cov_numer_denom = df[[numer_col, denom_col]].cov(ddof=1).iloc[0, 1]

    # Variances and covariance of the means
    var_mean_numer = var_numer / n
    var_mean_denom = var_denom / n
    cov_mean_numer_denom = cov_numer_denom / n

    # Delta method approximate variance
    ratio_variance = (mean_numer**2 / mean_denom**2) * (
        (var_mean_numer / mean_numer**2 + var_mean_denom / mean_denom**2) - 2 * cov_mean_numer_denom / (mean_numer * mean_denom)
    )

    # Normal approximation confidence interval
    ci_result = calc_standard_normal_ci(mean=ratio_estimate, se=np.sqrt(ratio_variance), ci_coverage=ci_coverage)

    return {
        "estimator_value": ratio_estimate,
        "estimator_var": ratio_variance,
        "ci": tuple(ci_result["ci"]),
    }
