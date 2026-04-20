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

import operator
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from abvelocity.core.stats.jackknife import calc_jk_stats
from abvelocity.core.stats.normal_ci import calc_standard_normal_ci
from abvelocity.core.stats.student_ci import calc_student_ci

# Define a type alias for clarity in dunder methods
EstimatorOrNumber = Union["Estimator", float, int]
# Add type alias for k-dim vector for inner product
EstimatorOrVector = Union["Estimator", np.ndarray]

# Map operator strings to safe functions
OPERATOR_MAP = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
}


@dataclass
class Estimator:
    """A statistical estimator (T) that is a function of observed random vectors X1,...,Xn.

    This class supports k-dimensional vector estimators, where k >= 1.
    The variance is a k x k variance-covariance matrix.
    Confidence Intervals (CI) are computed component-wise, providing marginal CIs.

    Attributes:
        value: The value of the estimator (k-dim vector).
        var: The estimated variance-covariance matrix (k x k).
        jk_value: The adjusted jackknife estimate (k-dim vector).
        jk_varcov: The estimated jackknife variance-covariance matrix (k x k).
        dof: Degrees of freedom.
        ci: Confidence interval (k x 2 matrix: [Lower, Upper] for k components).
        jk_ci: Jackknife Confidence interval (k x 2 matrix).
        current_value: Dynamic value for iterative methods (k-dim vector).
        param: Parameters used in estimation.
        true_mean: The true value of E(T) (k-dim vector), rarely known.
        true_varcov: The true value of Var(T) (k x k matrix), rarely known.
        name: An optional descriptive name for the estimator.
    """

    # --- K-Dimensional Attributes ---
    # k-dimensional vector (or 1D array for k=1)
    value: Optional[np.ndarray] = None
    jk_value: Optional[np.ndarray] = None
    current_value: Optional[np.ndarray] = None
    true_mean: Optional[np.ndarray] = None

    # k x k variance-covariance matrix
    var: Optional[np.ndarray] = None
    jk_varcov: Optional[np.ndarray] = None
    true_varcov: Optional[np.ndarray] = None

    # k x 2 matrix: (lower, upper) for each of the k components
    ci: Optional[np.ndarray] = None
    jk_ci: Optional[np.ndarray] = None

    # --- Univariate Attributes (Remain the same) ---
    dof: Optional[int] = None
    param: Optional[Any] = None
    name: Optional[str] = None

    @abstractmethod
    def estimator_func(self, df: pd.DataFrame, param: Optional[Any] = None) -> np.ndarray:
        """Abstract method to compute the k-dimensional estimator value.

        Args:
            df: The input DataFrame containing the data.
            param: Optional parameters for the estimator.

        Returns:
            The computed k-dimensional estimator value as a numpy array.
        """
        # NOTE: This must return a 1D numpy array of length k (k >= 1)
        pass

    def estimator_func_with_inferred_param(self, df: pd.DataFrame) -> np.ndarray:
        """Applies the estimator function with the stored param.

        Args:
            df: The input DataFrame containing the data.

        Returns:
            The computed estimator value as a numpy array using the stored param.
        """
        # This is the crucial entry point for execution used by calc_jk_stats
        return self.estimator_func(df=df, param=self.param)

    def calc_jk_stats(self, df: pd.DataFrame, num_buckets: int = 20, ci_coverage: float = 0.95) -> Dict[str, np.ndarray]:
        """Computes jackknife statistics and updates jk_value, jk_varcov, jk_ci, and dof attributes.

        This method leverages the k-dimensional nature of calc_jk_stats.

        Args:
            df: The input DataFrame containing the data.
            num_buckets: The number of groups (B) for jackknife iterations. Defaults to 20.
            ci_coverage: The desired coverage of the confidence interval, between 0 and 1. Defaults to 0.95.

        Returns:
            A dictionary containing estimator_value (k-dim), estimator_varcov (k x k), and ci (k x 2).
        """
        jk_res = calc_jk_stats(
            df=df,
            # The  estimator_func_with_inferred_param must return a k-dim numpy array
            estimator_func=self.estimator_func_with_inferred_param,
            num_buckets=num_buckets,
            ci_coverage=ci_coverage,
        )

        # The results from calc_jk_stats are already k-dimensional numpy arrays
        self.jk_value = jk_res["estimator_value"]
        self.jk_varcov = jk_res["estimator_varcov"]
        self.jk_ci = jk_res["ci"]
        self.dof = num_buckets - 1

        return jk_res

    def calc_normal_ci(self, ci_coverage: float = 0.95) -> np.ndarray:
        """Calculates the MARGINAL confidence intervals using the normal approximation and updates ci.

        It applies the CI calculation component-wise using the diagonal elements (marginal variances)
        of the VarCov matrix.

        Args:
            ci_coverage: The desired coverage of the confidence interval, between 0 and 1. Defaults to 0.95.

        Returns:
            A k x 2 numpy array containing the lower and upper bounds for each component.

        Raises:
            ValueError: If value or var is None or if var has non-positive marginal variances.
        """
        if self.value is None or self.var is None:
            raise ValueError("Cannot calculate normal CI: 'value' or 'var' is None.")

        # Determine k.
        if self.value.ndim != 1:
            raise ValueError("'value' must be a 1D array for CI calculation.")

        k = self.value.shape[0]
        ci_matrix = np.empty((k, 2))

        # Iterate over each component i=0, 1, ..., k-1
        for i in range(k):
            mean_i = self.value[i]
            # Use the MARGINAL variance (diagonal element) for the standard error
            # If k=1 (e.g., after an inner product), self.var is 1x1, self.var[0, 0] is the only element.
            if self.var.shape == (1, 1):
                var_i = self.var[0, 0]
            else:
                var_i = self.var[i, i]

            if var_i <= 0:
                raise ValueError(f"Cannot calculate normal CI: Component {i} variance is non-positive ({var_i}).")

            ci_result = calc_standard_normal_ci(mean=mean_i, se=np.sqrt(var_i), ci_coverage=ci_coverage)
            ci_matrix[i, :] = ci_result["ci"]

        self.ci = ci_matrix
        return self.ci

    def calc_student_ci(self, ci_coverage: float = 0.95) -> np.ndarray:
        """Calculates the MARGINAL confidence intervals using Student's t-distribution and updates ci.

        It applies the CI calculation component-wise using the diagonal elements (marginal variances)
        of the VarCov matrix.

        Args:
            ci_coverage: The desired coverage of the confidence interval, between 0 and 1. Defaults to 0.95.

        Returns:
            A k x 2 numpy array containing the lower and upper bounds for each component.

        Raises:
            ValueError: If value, var, or dof is None, or if var has non-positive marginal variances, or if dof is not positive.
        """
        if self.value is None or self.var is None or self.dof is None:
            raise ValueError("Cannot calculate student CI: 'value', 'var', or 'dof' is None.")
        if self.dof <= 0:
            raise ValueError(f"Cannot calculate student CI: 'dof' must be positive ({self.dof}).")

        # Determine k.
        if self.value.ndim != 1:
            raise ValueError("'value' must be a 1D array for CI calculation.")

        k = self.value.shape[0]
        ci_matrix = np.empty((k, 2))

        # Iterate over each component i=0, 1, ..., k-1
        for i in range(k):
            mean_i = self.value[i]
            # Use the MARGINAL variance (diagonal element) for the standard error
            if self.var.shape == (1, 1):
                var_i = self.var[0, 0]
            else:
                var_i = self.var[i, i]

            if var_i <= 0:
                raise ValueError(f"Cannot calculate student CI: Component {i} variance is non-positive ({var_i}).")

            ci_result = calc_student_ci(mean=mean_i, se=np.sqrt(var_i), dof=self.dof, ci_coverage=ci_coverage)
            ci_matrix[i, :] = ci_result["ci"]

        self.ci = ci_matrix
        return self.ci

    def _get_operand_name(self, operand: EstimatorOrVector) -> str:
        """Helper to get the name of the operand for composite name generation."""
        if isinstance(operand, Estimator):
            return operand.name if operand.name else "Estimator"

        # For a k-dim vector
        if isinstance(operand, np.ndarray):
            return f"Vector({operand.shape[0]})"

        return str(operand)

    def _create_composite(self, operand1: EstimatorOrNumber, operand2: EstimatorOrNumber, operator_str: str) -> "Estimator":
        """
        Creates and returns a new Estimator instance representing the element-wise arithmetic operation.
        """
        name_operand1 = self._get_operand_name(operand1)
        name_operand2 = self._get_operand_name(operand2)
        new_name = f"({name_operand1} {operator_str} {name_operand2})"

        op_func = OPERATOR_MAP.get(operator_str)
        if op_func is None:
            raise NotImplementedError(f"Operator '{operator_str}' not supported.")

        if operator_str == "/" and not isinstance(operand2, Estimator) and operand2 == 0:
            raise ZeroDivisionError("Cannot divide by zero constant.")

        new_est = Estimator(name=new_name)

        # --- Functional Composition Logic (Closure over operands) ---
        if isinstance(operand1, Estimator) and isinstance(operand2, Estimator):
            # Estimator op Estimator: Element-wise operation on k-dim results
            new_est.estimator_func = lambda df, param: op_func(
                operand1.estimator_func_with_inferred_param(df),
                operand2.estimator_func_with_inferred_param(df),
            )
        elif isinstance(operand1, Estimator):
            # Estimator op Number: Scalar is applied to all k elements
            new_est.estimator_func = lambda df, param: op_func(operand1.estimator_func_with_inferred_param(df), operand2)
        else:  # Number op Estimator: Scalar is applied to all k elements
            new_est.estimator_func = lambda df, param: op_func(operand1, operand2.estimator_func_with_inferred_param(df))

        return new_est

    def _create_inner_product_composite(self, operand1: EstimatorOrVector, operand2: EstimatorOrVector) -> "Estimator":
        """
        Creates and returns a new 1-D (scalar) Estimator representing the inner product.
        Inner product: operand1.T @ operand2
        """
        name_operand1 = self._get_operand_name(operand1)
        name_operand2 = self._get_operand_name(operand2)
        new_name = f"({name_operand1} @ {name_operand2})"

        # Inner product always results in a 1-dimensional estimator (a scalar)
        new_est = Estimator(name=new_name)

        # --- Functional Composition Logic for Inner Product ---
        if isinstance(operand1, Estimator) and isinstance(operand2, Estimator):
            # Estimator @ Estimator: T1^T @ T2 (returns a 1-element array)
            new_est.estimator_func = lambda df, param: np.array(
                [
                    np.inner(
                        operand1.estimator_func_with_inferred_param(df),
                        operand2.estimator_func_with_inferred_param(df),
                    )
                ]
            )

        elif isinstance(operand1, Estimator):
            # Estimator @ Vector: T^T @ w
            if not isinstance(operand2, np.ndarray) or operand2.ndim != 1:
                raise TypeError("Right operand for inner product must be an Estimator or a 1D numpy array.")
            # The lambda closes over operand2 (numpy vector)
            new_est.estimator_func = lambda df, param: np.array([np.inner(operand1.estimator_func_with_inferred_param(df), operand2)])

        elif isinstance(operand2, Estimator):
            # Vector @ Estimator: w^T @ T
            if not isinstance(operand1, np.ndarray) or operand1.ndim != 1:
                raise TypeError("Left operand for inner product must be an Estimator or a 1D numpy array.")
            # The lambda closes over operand1 (numpy vector)
            new_est.estimator_func = lambda df, param: np.array([np.inner(operand1, operand2.estimator_func_with_inferred_param(df))])

        else:
            raise TypeError("Inner product requires at least one operand to be an Estimator.")

        return new_est

    # --- DUNDER METHODS (Element-wise arithmetic) ---

    def __add__(self, other: EstimatorOrNumber) -> "Estimator":
        return self._create_composite(self, other, "+")

    def __radd__(self, other: EstimatorOrNumber) -> "Estimator":
        return self._create_composite(other, self, "+")

    def __sub__(self, other: EstimatorOrNumber) -> "Estimator":
        return self._create_composite(self, other, "-")

    def __rsub__(self, other: EstimatorOrNumber) -> "Estimator":
        return self._create_composite(other, self, "-")

    def __mul__(self, other: EstimatorOrNumber) -> "Estimator":
        return self._create_composite(self, other, "*")

    def __rmul__(self, other: EstimatorOrNumber) -> "Estimator":
        return self._create_composite(other, self, "*")

    def __truediv__(self, other: EstimatorOrNumber) -> "Estimator":
        return self._create_composite(self, other, "/")

    def __rtruediv__(self, other: EstimatorOrNumber) -> "Estimator":
        return self._create_composite(other, self, "/")

    # --- DUNDER METHOD for Inner Product (@) ---

    def __matmul__(self, other: EstimatorOrVector) -> "Estimator":
        """Implements the inner product (matrix multiplication) logic (T1 @ T2 or T @ w)."""
        return self._create_inner_product_composite(self, other)

    def __rmatmul__(self, other: EstimatorOrVector) -> "Estimator":
        """Implements the reverse inner product (w @ T). Note: Inner product is commutative."""
        return self._create_inner_product_composite(other, self)
