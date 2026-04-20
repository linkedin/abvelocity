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
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# Original author: Reza Hosseini

import numpy as np
from scipy.stats import power_divergence


def stat_test_compare_distbns(distbns, true_distbn=None, method="LR"):
    """
    Performs statistical tests to compare multinomial distributions using either
    Pearson's Chi-squared or Likelihood Ratio (G-test) statistics.

    The function handles two distinct statistical settings:

    1. Goodness-of-Fit (GoF):
       Tests a simple null hypothesis where the observed counts are compared against
       a fully specified probability vector (true_distbn).
       Degrees of Freedom = K - 1 (for a single row) or R*(K-1) (for multiple rows
       tested against the same fixed baseline).
       Reference: Agresti, A. (2002). Categorical Data Analysis. Section 2.1.

    2. Test of Homogeneity:
       Tests a composite null hypothesis where multiple independent multinomial
       samples are compared to see if they share a common (unspecified) parameter
       vector. The expected values are estimated from the marginal totals.
       Degrees of Freedom = (R - 1) * (K - 1).
       Reference: Lehmann, E. L., & Romano, J. P. (2005). Testing Statistical Hypotheses.

    Args:
        distbns (list of tuples/lists): Observed counts. Each element is a distribution.
        true_distbn (tuple/list, optional): The fixed probability vector for GoF tests.
        method (str): "LR" for Likelihood Ratio (G-test) or "Pearson" for Chi-squared.

    Returns:
        dict: Contains method, statistic, p_value, dof, and test_type.
    """
    if not distbns:
        raise ValueError("The distbns list is empty.")

    sizes = [len(d) for d in distbns]
    if true_distbn is not None:
        sizes.append(len(true_distbn))

    if len(set(sizes)) > 1:
        raise ValueError(f"All distributions must have the same number of categories. Found sizes: {sizes}")

    obs = np.array(distbns)

    if method == "LR":
        lambda_val = "log-likelihood"
    elif method == "Pearson":
        lambda_val = 1
    else:
        raise ValueError("Method must be 'LR' or 'Pearson'")

    if true_distbn is not None:
        true_arr = np.array(true_distbn)
        true_probs = true_arr / true_arr.sum()

        row_totals = obs.sum(axis=1, keepdims=True)
        expected = row_totals * true_probs

        stat, p_val = power_divergence(f_obs=obs.ravel(), f_exp=expected.ravel(), lambda_=lambda_val)

        dof = obs.shape[0] * (obs.shape[1] - 1)
        test_type = "Goodness-of-Fit"
    else:
        stat, p_val = power_divergence(obs, lambda_=lambda_val, axis=None)

        dof = (obs.shape[0] - 1) * (obs.shape[1] - 1)
        test_type = "Homogeneity"

    return {
        "method": method,
        "statistic": stat,
        "p_value": p_val,
        "dof": dof,
        "test_type": test_type,
    }
