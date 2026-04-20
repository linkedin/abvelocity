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
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# Original author: Reza Hosseini

from typing import Any, Callable, Dict, Union

import numpy as np
import pandas as pd
from abvelocity.core.stats.student_ci import calc_student_ci

# Define the type for the estimator function (must return a numpy array/vector)
EstimatorFuncND = Callable[[pd.DataFrame], np.ndarray]


def calc_jk_stats(
    df: pd.DataFrame,
    estimator_func: EstimatorFuncND,
    num_buckets: int = 20,
    ci_coverage: float = 0.95,
) -> Dict[str, Union[np.ndarray, Any]]:
    """
    Computes the bias-corrected grouped jackknife estimate, its variance-covariance matrix,
    confidence interval(s), and p-values. This function is designed for k-dimensional output.

    The output is always in the k-dimensional format: a vector for the estimate,
    a covariance matrix for variance, a k x 2 array for CIs, and a vector for p-values,
    even if k=1.

    Args:
    - df (pd.DataFrame): The full dataset.
    - estimator_func (Callable[[pd.DataFrame], np.ndarray]): The k-dimensional estimator function.
      Must return a scalar or a 1D numpy array.
    - num_buckets (int): The number of groups (B) and the number of jk iterations. Default is 20.
    - ci_coverage (float): The desired coverage of the confidence interval, a value between 0 and 1. Default is 0.95.

    Returns:
    - dict[str, Union[np.ndarray, Any]]: A dictionary containing:
      - "estimator_value": The bias-corrected jackknife estimate vector (1D array, shape (k,)).
      - "estimator_varcov": The jackknife variance-covariance matrix (2D array, shape (k, k)).
      - "ci": A numpy array of shape (k, 2), representing the lower and upper bounds for each of the k CIs.
      - "p_values": A numpy array of shape (k,), containing two-sided p-values for each dimension under the null hypothesis that the true parameter is zero.

    Raises:
    - ValueError: If num_buckets or ci_coverage is invalid.
    - RuntimeError: If the estimator function does not return a scalar or a 1D numpy array.
    """
    n = len(df)

    if num_buckets <= 1 or num_buckets > n:
        raise ValueError("num_buckets must be an integer between 2 and the df size (n).")

    if not (0 < ci_coverage < 1):
        raise ValueError(f"ci_coverage has to be between 0 and 1: {ci_coverage}")

    # 1. Full sample estimate ($\hat{\theta}$)
    theta_hat_raw = estimator_func(df)

    # Coerce any scalar output to a 1D numpy array (k=1 vector)
    if np.isscalar(theta_hat_raw):
        theta_hat = np.array([theta_hat_raw])
    elif isinstance(theta_hat_raw, np.ndarray) and theta_hat_raw.ndim == 1:
        theta_hat = theta_hat_raw
    else:
        raise RuntimeError("The estimator function must return a scalar or a 1D numpy array (vector).")

    k = theta_hat.shape[0]  # The dimension of the estimator output

    # Determine bucket size and array for jk estimates
    bucket_size = int(np.ceil(n / num_buckets))
    # jk_estimates is an array of shape (B, k)
    jk_estimates = np.empty((num_buckets, k), dtype=float)

    # Generate the full set of indices (0 to n-1)
    full_indices = np.arange(n)

    # 2. Compute the jackknife samples ($\hat{\theta}_{(i)}$)
    for i in range(num_buckets):
        # print(f"\n***\n: {i}-th iteration of jackknife out of {num_buckets}")
        # Determine the start and end indices for the current bucket to be removed
        start_idx = i * bucket_size
        end_idx = min(start_idx + bucket_size, n)

        # Create indices to keep by removing the slice [start_idx:end_idx] from the full set
        indices_to_keep = np.delete(full_indices, np.s_[start_idx:end_idx])

        # Create 'leave-one-bucket-out' subset
        jk_subset = df.iloc[indices_to_keep]

        # Calculate estimate for the subset and coerce to vector if needed
        jk_estimate_i = estimator_func(jk_subset)
        if np.isscalar(jk_estimate_i):
            jk_estimates[i, :] = np.array([jk_estimate_i])
        else:
            jk_estimates[i, :] = jk_estimate_i

    # 3. Calculate mean of jk estimates ($\bar{\hat{\theta}}$)
    theta_bar = np.mean(jk_estimates, axis=0)

    # --- Jackknife Variance-Covariance Matrix ($\mathbf{V}_{JK}$) ---
    # Centered jk estimates: $\hat{\theta}_{(i)} - \bar{\hat{\theta}}$ (shape (B, k))
    centered_jk_estimates = jk_estimates - theta_bar

    # Compute $\mathbf{C}_{sum} = \sum (\hat{\theta}_{(i)} - \bar{\hat{\theta}})(\hat{\theta}_{(i)} - \bar{\hat{\theta}})^T$
    C_sum = centered_jk_estimates.T @ centered_jk_estimates

    var_factor = num_buckets - 1
    estimator_varcov = (var_factor / num_buckets) * C_sum

    # --- Bias-Corrected Jackknife Estimate ($\hat{\theta}_{JK}$) ---
    # Jackknife Bias ($\text{Bias}_{JK}$) scaled by B
    jk_bias = (num_buckets - 1) * (theta_bar - theta_hat)

    # Bias-corrected estimate
    estimator_value = theta_hat - jk_bias

    # --- Confidence Intervals and P-values using Student's t-distribution ---
    dof = num_buckets - 1

    # We calculate k independent CIs and p-values using the diagonal elements (standard errors)
    ci_array = np.empty((k, 2), dtype=float)
    p_values = np.empty(k, dtype=float)
    std_errors = np.sqrt(np.diag(estimator_varcov))

    for j in range(k):
        # Confidence interval
        ci_result = calc_student_ci(mean=estimator_value[j], se=std_errors[j], dof=dof, ci_coverage=ci_coverage)
        ci_array[j, :] = ci_result["ci"]
        p_values[j] = ci_result["p_value"]

    # Return the k-dimensional format for all cases (k >= 1)
    return {
        "estimator_value": estimator_value,  # 1D array (k,)
        "estimator_varcov": estimator_varcov,  # 2D array (k, k)
        "ci": ci_array,  # 2D array (k, 2)
        "p_values": p_values,  # 1D array (k,)
    }
