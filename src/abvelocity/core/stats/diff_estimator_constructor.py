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

from typing import Any, Optional

import numpy as np
import pandas as pd
from abvelocity.core.stats.estimator import Estimator


class DiffEstimatorConstructor:
    """A class to construct Estimators for comparing control and treatment estimators.

    Initializes with a diff_type and optional name, then constructs an Estimator via the construct method.

    Attributes:
        diff_type: Type of difference to compute: "simple_diff", "pcnt_diff", or "both".
        name: Optional custom name for the constructed Estimator.
    """

    def __init__(self, diff_type: str = "both", name: Optional[str] = None):
        """Initialize with the type of difference and optional name.

        Args:
            diff_type: Type of difference: "simple_diff", "pcnt_diff", or "both" (default).
            name: Optional custom name for the constructed Estimator (default None).

        Raises:
            ValueError: If diff_type is invalid.
        """
        valid_diff_types = {"simple_diff", "pcnt_diff", "both"}
        if diff_type not in valid_diff_types:
            raise ValueError(f"diff_type must be one of {valid_diff_types}, got {diff_type}.")
        self.diff_type = diff_type
        self.name = name

    def construct(self, control_estimator: Estimator, treatment_estimator: Estimator) -> Estimator:
        """Constructs an Estimator computing element-wise difference and/or percentage change.

        For two k-dimensional estimators T1 (control) and T2 (treatment), returns an estimator with:
        - diff_type="simple_diff": k-dimensional output with T2 - T1 (treatment minus control).
        - diff_type="pcnt_diff": k-dimensional output with ((T2 - T1) / T1) * 100.
        - diff_type="both": 2k-dimensional output with [T2 - T1, ((T2 - T1) / T1) * 100].
        Returns NaN for percentage changes where T1's components are zero.

        Args:
            control_estimator: The control Estimator (T1).
            treatment_estimator: The treatment Estimator (T2).

        Returns:
            A new Estimator with k-dimensional (simple_diff or pcnt_diff) or 2k-dimensional (both) output.

        Raises:
            TypeError: If either input is not an Estimator.
            ValueError: If T1 and T2 have different dimensions.
        """
        if not isinstance(control_estimator, Estimator) or not isinstance(treatment_estimator, Estimator):
            raise TypeError("Both control_estimator and treatment_estimator must be Estimator objects.")

        name_control = control_estimator.name if control_estimator.name else "Control"
        name_treatment = treatment_estimator.name if treatment_estimator.name else "Treatment"
        if self.name is not None:
            new_name = self.name
        elif self.diff_type == "simple_diff":
            new_name = f"({name_treatment} - {name_control})"
        elif self.diff_type == "pcnt_diff":
            new_name = f"(({name_treatment} - {name_control})/{name_control} * 100)"
        else:  # both
            new_name = f"({name_treatment} - {name_control}, ({name_treatment} - {name_control})/{name_control} * 100)"

        new_est = Estimator(name=new_name)

        def estimator_func(df: pd.DataFrame, param: Optional[Any] = None) -> np.ndarray:
            t1 = control_estimator.estimator_func_with_inferred_param(df)
            t2 = treatment_estimator.estimator_func_with_inferred_param(df)
            if t1.shape != t2.shape:
                raise ValueError(f"Estimators must have the same dimension: {t1.shape} vs {t2.shape}.")
            diff = t2 - t1  # Treatment minus control
            if self.diff_type == "simple_diff":
                return diff

            with np.errstate(divide="warn", invalid="warn"):
                percent_change = np.where(t1 != 0, (diff / t1) * 100, np.nan)
            if self.diff_type == "pcnt_diff":
                return percent_change
            return np.concatenate([diff, percent_change])  # both

        new_est.estimator_func = estimator_func
        return new_est
