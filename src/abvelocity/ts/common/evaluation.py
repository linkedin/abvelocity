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
# original author: Reza Hosseini, Albert Chen
"""Evaluation functions.
Valid input processing is done within these evaluation functions
so they can be called from anywhere without error checks.
"""

import math
import warnings
from collections import namedtuple
from enum import Enum
from functools import partial
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from abvelocity.ts.common.constants import (
    ACTUAL_COL,
    COVERAGE_VS_INTENDED_DIFF,
    LOWER_BAND_COVERAGE,
    PREDICTED_COL,
    PREDICTED_LOWER_COL,
    PREDICTED_UPPER_COL,
    PREDICTION_BAND_COVERAGE,
    PREDICTION_BAND_WIDTH,
    UPPER_BAND_COVERAGE,
)
from abvelocity.ts.common.logging import LoggingLevelEnum, log_message
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score


def all_equal_length(*arrays):
    """Checks whether input arrays have equal length.

    Args:
        arrays: one or more lists/numpy arrays/pd.Series.

    Returns:
        True if lengths are all equal, otherwise False.
    """
    length = None
    for arr in arrays:
        try:
            if length is not None and len(arr) != length:
                return False
            length = len(arr)
        except TypeError:  # continue if `arr` has no len method. constants and None are excluded from length check
            pass
    return True


def valid_elements_for_evaluation(
    reference_arrays: List[Union[np.array, pd.Series, List[Union[int, float]]]],
    arrays: List[Optional[Union[int, float, np.array, pd.Series, List[Union[int, float]]]]],
    reference_array_names: str,
    drop_leading_only: bool,
    keep_inf: bool,
):
    """Keeps finite elements from reference_array, and corresponding elements in *arrays.

    Args:
        reference_arrays: The reference arrays where the indices of NA/infs are populated.
            If length is longer than 1, a logical and will be used to choose valid elements.
            ``list`` [``numpy.array``, ``pandas.Series`` or ``list`` [``int``, ``float``]].
        arrays: The arrays with the indices of NA/infs in ``reference_array`` to be dropped.
            ``list`` [``int``, ``float``, ``numpy.array``, ``pandas.Series`` or ``list`` [``int``, ``float``]].
        reference_array_names: The reference array name to be printed in the warning.  ``str``.
        drop_leading_only: True means dropping the leading NA/infs only
            (drop the leading indices whose values are not valid in any reference array).
            False means dropping all NA/infs regardless of where they are.  ``bool``.
        keep_inf: True means dropping NA only.
            False means dropping both NA and INF.  ``bool``.

    Returns:
        arrays_valid_elements: List of numpy arrays with valid indices [*reference_arrays, *arrays].
            ``list`` [``numpy.array``].
    """
    if not all_equal_length(*reference_arrays, *arrays):
        raise ValueError("length of arrays do not match")

    if len(reference_arrays) == 0:
        return reference_arrays + arrays

    reference_arrays = [np.array(reference_array) for reference_array in reference_arrays]
    array_length = reference_arrays[0].shape[0]

    # Defines a function to perform the opposite of `numpy.isnan`.
    def is_not_nan(x):
        """Gets the True/False for elements that are not/are NANs.

        Args:
            x: The input array.  array-like.

        Returns:
            is_not_nan: True/False array indicating whether the elements are not NAN/are NAN.  ``numpy.array``.
        """
        return ~np.isnan(x)

    validation_func = is_not_nan if keep_inf else np.isfinite
    # Finds the indices of finite elements in reference array
    keep = [validation_func(reference_array) for reference_array in reference_arrays]
    if drop_leading_only:
        # Gets the index where the first True is.
        # All the False after this True will not be heading False
        # and shouldn't be dropped.
        # If multiple arrays in ``reference_arrays``,
        # this will be the minimum.
        valid_indices = [np.where(array)[0] for array in keep]
        heading_lengths = [length[0] if length.shape[0] > 0 else array_length for length in valid_indices]
        heading_length = min(heading_lengths)
        # Generates arrays with the is_heading flag.
        is_not_heading = np.repeat([False, True], [heading_length, array_length - heading_length])
    else:
        is_not_heading = np.repeat([False, True], [array_length, 0])
    keep = np.logical_and.reduce(keep, axis=0)
    keep = np.logical_or(keep, is_not_heading)
    num_remaining = keep.sum()
    num_removed = array_length - num_remaining
    removed_elements = "NA" if keep_inf else "NA or infinite"
    if num_remaining == 0:
        warnings.warn("There are 0 non-null elements for evaluation.")
    if num_removed > 0:
        warnings.warn(f"{num_removed} value(s) in {reference_array_names} were {removed_elements} and are omitted in error calc.")

    # Keeps these indices in all arrays. Leaves float, int, and None as-is
    return [array[keep] for array in reference_arrays] + [
        (np.array(array)[keep] if (array is not None and not isinstance(array, (float, int, np.float32))) else array) for array in arrays
    ]


def aggregate_array(ts_values, agg_periods=7, agg_func=np.sum):
    """Aggregates input array.

    Divides array from left to right into bins of size agg_periods, and applies agg_func to each block.
    Drops records from the left if needed to ensure all bins are full.

    Args:
        ts_values: list, np.array, or pd.Series to aggregate.
        agg_periods: number of periods to combine in aggregation.
        agg_func: aggregation function, e.g. np.max, np.sum. Must take an array and return a number.

    Returns:
        array, aggregated so that every agg_periods periods are combined into one.

    Examples:
        >>> aggregate_array([1.0, 2.0, 3.0, 4.0], 2, np.sum)
        array([3., 7.])
        >>> aggregate_array(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]), 2, np.sum)
        array([5., 9.])
        >>> aggregate_array(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 2, np.max)
        array([3., 5.])
    """
    ts_values = np.array(ts_values)
    n_periods = len(ts_values)

    drop_first_periods = n_periods % agg_periods  # drop these periods from the front, to ensure all bins are full
    if drop_first_periods == n_periods:
        drop_first_periods = 0
        warnings.warn(f"Requested agg_periods={agg_periods}, but there are only {n_periods}. Using all for aggregation")
    elif drop_first_periods > 0:
        log_message(
            f"Requested agg_periods={agg_periods} for data of length {n_periods}. Dropping first" f" {drop_first_periods} records before aggregation",
            LoggingLevelEnum.INFO,
        )

    # creates dummy time index for aggregation
    dates = pd.date_range("2018-01-01", periods=n_periods - drop_first_periods, freq="1D")
    ts = pd.Series(ts_values[drop_first_periods:], index=dates)
    aggregated_array = ts.resample(f"{agg_periods}D", closed="left").agg(lambda x: agg_func(x)).values
    return aggregated_array


def add_preaggregation_to_scorer(score_func, agg_periods=7, agg_func=np.sum):
    """Takes a scorer and returns a scorer that pre-aggregates input.

    Args:
        score_func: function that maps two arrays to a number. E.g. (y_true, y_pred) -> error.
        agg_periods: number of periods to combine in aggregation.
        agg_func: function that takes an array and returns a number, e.g. np.max, np.sum. This function should
            handle np.nan without error.

    Returns:
        scorer that applies agg_func to data before calling score_func.

    For example, if have daily data and want to evaluate MSE on weekly totals, the appropriate scorer is:
        add_preaggregation_to_scorer(mean_squared_error, agg_periods=7, agg_func=np.max)
    """

    def score_func_preagg(y_true, y_pred, **kwargs):
        y_true_agg = aggregate_array(y_true, agg_periods, agg_func)
        y_pred_agg = aggregate_array(y_pred, agg_periods, agg_func)
        return score_func(y_true_agg, y_pred_agg, **kwargs)

    return score_func_preagg


def add_finite_filter_to_scorer(score_func):
    """Takes a scorer and returns a scorer that ignores NA / infinite elements in y_true.

    sklearn scorers (and others) don't handle arrays with 0 length. In that case, return None.

    Args:
        score_func: function that maps two arrays to a number. E.g. (y_true, y_pred) -> error.

    Returns:
        scorer that drops records where y_true is not finite.
    """

    def score_func_finite(y_true, y_pred, **kwargs):
        y_true, y_pred = valid_elements_for_evaluation(
            reference_arrays=[y_true],
            arrays=[y_pred],
            reference_array_names="y_true",
            drop_leading_only=False,
            keep_inf=False,
        )
        # The Silverkite Multistage model has NANs at the beginning
        # when predicting on the training data.
        # We only drop the leading NANs/infs from ``y_pred``,
        # since they are not supposed to appear in the middle.
        y_pred, y_true = valid_elements_for_evaluation(
            reference_arrays=[y_pred],
            arrays=[y_true],
            reference_array_names="y_pred",
            drop_leading_only=True,
            keep_inf=True,
        )
        if len(y_true) == 0:  # returns None if there are no elements
            return None
        return score_func(y_true, y_pred, **kwargs)

    return score_func_finite


def r2_null_model_score(y_true, y_pred, y_pred_null=None, y_train=None, loss_func=mean_squared_error):
    """Calculates improvement in the loss function compared to the predictions of a null model.

    Can be used to evaluate model quality with respect to a simple baseline model.

    The score is defined as::

        R2_null_model_score = 1.0 - loss_func(y_true, y_pred) / loss_func(y_true, y_pred_null)

    Args:
        y_true: Observed response (usually on a test set).  ``list`` [``float``] or ``numpy.array``.
        y_pred: Model predictions (usually on a test set).  ``list`` [``float``] or ``numpy.array``.
        y_pred_null: A baseline prediction model to compare against.
            If None, derived from ``y_train`` or ``y_true``.
            ``list`` [``float``] or ``numpy.array`` or None.
        y_train: Response values in the training data.
            If ``y_pred_null`` is None, then ``y_pred_null`` is set to the mean of ``y_train``.
            If ``y_train`` is also None, then ``y_pred_null`` is set to the mean of ``y_true``.
            ``list`` [``float``] or ``numpy.array`` or None.
        loss_func: The error loss function with signature (true_values, predicted_values).
            callable, default ``sklearn.metrics.mean_squared_error``.

    Returns:
        r2_null_model: A value within (-\\infty, 1.0]. Higher scores are better.  ``float``.
            Can be interpreted as the improvement in the loss function
            compared to the predictions of the null model.
            For example, a score of 0.74 means the loss is 74% lower
            than for the null model.

    Notes:
        There is a connection between ``R2_null_model_score`` and ``R2``.
        ``R2_null_model_score`` can be interpreted as the additional improvement in the
        coefficient of determination (i.e. ``R2``, see `sklearn.metrics.r2_score`) with
        respect to a null model.

        Under the default settings of this function, where ``loss_func`` is
        mean squared error and ``y_pred_null`` is the average of
        ``y_true``, the scores are equivalent::

            # simplified definition of R2_score, where SSE is sum of squared error
            y_true_avg = np.repeat(np.average(y_true), y_true.shape[0])
            R2_score := 1.0 - SSE(y_true, y_pred) / SSE(y_true, y_true_avg)
            R2_score := 1.0 - MSE(y_true, y_pred) / VAR(y_true)  # equivalent definition

            r2_null_model_score(y_true, y_pred) == r2_score(y_true, y_pred)

        ``r2_score`` is 0 if simply predicting the mean (y_pred = y_true_avg).

        If ``y_pred_null`` is passed, and if ``loss_func`` is mean squared error
        and ``y_true`` has nonzero variance, this function measures how much "r2_score of the
        predictions (``y_pred``)" closes the gap between "r2_score of the null model (``y_pred``)"
        and the "r2_score of the best possible model (``y_true``)", which is 1.0::

            R2_pred = r2_score(y_true, y_pred)       # R2 of predictions
            R2_null = r2_score(y_pred_null, y_pred)  # R2 of null model
            r2_null_model_score(y_true, y_pred, y_pred_null) == (R2_pred - R2_null) / (1.0 - R2_null)

        When ``y_pred_null=y_true_avg``, ``R2_null`` is 0 and this reduces to the formula above.

        Summary (for ``loss_func=mean_squared_error``):

            - If ``R2_null>0`` (good null model), then ``R2_null_model_score < R2_score``
            - If ``R2_null=0`` (uninformative null model), then ``R2_null_model_score = R2_score``
            - If ``R2_null<0`` (poor null model), then ``R2_null_model_score > R2_score``

        For other loss functions, ``r2_null_model_score`` has the same connection to pseudo R2.
    """
    y_true, y_pred, y_train, y_pred_null = valid_elements_for_evaluation(
        reference_arrays=[y_true],
        arrays=[y_pred, y_train, y_pred_null],
        reference_array_names="y_true",
        drop_leading_only=False,
        keep_inf=False,
    )
    r2_null_model = None

    if len(y_true) > 0:
        # determines null model
        if y_pred_null is None and y_train is not None:  # from training data
            y_pred_null = np.repeat(y_train.mean(), len(y_true))
        elif y_pred_null is None:  # from test data
            y_pred_null = np.repeat(y_true.mean(), len(y_true))
        elif isinstance(y_pred_null, (float, int, np.float32)):  # constant null model
            y_pred_null = np.repeat(y_pred_null, len(y_true))
        # otherwise, y_pred_null is an array, used directly

        model_loss = loss_func(y_true, y_pred)
        null_loss = loss_func(y_true, y_pred_null)

        if null_loss > 0.0:
            r2_null_model = 1.0 - (model_loss / null_loss)
    return r2_null_model


def calc_pred_err(y_true, y_pred):
    """Calculates the basic error measures in
    `~abvelocity.ts.common.evaluation.EvaluationMetricEnum`
    and returns them in a dictionary.

    Args:
        y_true: Observed values.  ``list`` [``float``] or ``numpy.array``.
        y_pred: Model predictions.  ``list`` [``float``] or ``numpy.array``.

    Returns:
        error: Dictionary mapping
            `~abvelocity.ts.common.evaluation.EvaluationMetricEnum`
            metric names to their values for the given ``y_true`` and ``y_pred``.
            ``dict`` [``str``, ``float`` or None].

            The dictionary is empty if there are no finite elements
            in ``y_true``.
    """
    y_true, y_pred = valid_elements_for_evaluation(
        reference_arrays=[y_true],
        arrays=[y_pred],
        reference_array_names="y_true",
        drop_leading_only=False,
        keep_inf=False,
    )
    # The Silverkite Multistage model has NANs at the beginning
    # when predicting on the training data.
    # We only drop the leading NANs/infs from ``y_pred``,
    # since they are not supposed to appear in the middle.
    y_pred, y_true = valid_elements_for_evaluation(
        reference_arrays=[y_pred],
        arrays=[y_true],
        reference_array_names="y_pred",
        drop_leading_only=True,
        keep_inf=True,
    )
    error = {}

    if len(y_true) > 0:
        for enum in EvaluationMetricEnum:
            metric_name = enum.get_metric_name()
            metric_func = enum.get_metric_func()
            error.update({metric_name: metric_func(y_true, y_pred)})
    return error


@add_finite_filter_to_scorer
def mean_absolute_percent_error(y_true, y_pred):
    """Calculates mean absolute percentage error.

    Args:
        y_true: observed values given in a list (or numpy array).
        y_pred: predicted values given in a list (or numpy array).

    Returns:
        mean absolute percent error.
    """
    smallest_denominator = np.min(np.abs(y_true))
    if smallest_denominator == 0:
        warnings.warn("y_true contains 0. MAPE is undefined.")
        return None
    elif smallest_denominator < 1e-8:
        warnings.warn("y_true contains very small values. MAPE is likely highly volatile.")
    return 100 * (np.abs(y_true - y_pred) / np.abs(y_true)).mean()


@add_finite_filter_to_scorer
def median_absolute_percent_error(y_true, y_pred):
    """Calculates median absolute percentage error.

    Args:
        y_true: observed values given in a list (or numpy array).
        y_pred: predicted values given in a list (or numpy array).

    Returns:
        median absolute percent error.
    """
    smallest_denominator = np.min(np.abs(y_true))
    if smallest_denominator == 0:
        warnings.warn("y_true contains 0. MedAPE is undefined.")
        return None
    elif smallest_denominator < 1e-8:
        num_small = len(y_true[y_true < 1e-8])
        if num_small > (len(y_true) - 1) // 2:
            # if too many very small values, median is affected
            warnings.warn("y_true contains very small values. MedAPE is likely highly volatile.")
    return 100 * np.median((np.abs(y_true - y_pred) / np.abs(y_true)))


@add_finite_filter_to_scorer
def symmetric_mean_absolute_percent_error(y_true, y_pred):
    """Calculates symmetric mean absolute percentage error.

    Note that we do not include a factor of 2 in the denominator, so the range is 0% to 100%.

    Args:
        y_true: observed values given in a list (or numpy array).
        y_pred: predicted values given in a list (or numpy array).

    Returns:
        symmetric mean absolute percent error.
    """
    smallest_denominator = np.min(np.abs(y_true) + np.abs(y_pred))
    if smallest_denominator == 0:
        warnings.warn("denominator contains 0. sMAPE is undefined.")
        return None
    elif smallest_denominator < 1e-8:
        warnings.warn("denominator contains very small values. sMAPE is likely highly volatile.")
    return 100 * (np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))).mean()


@add_finite_filter_to_scorer
def root_mean_squared_error(y_true, y_pred):
    """Calculates root mean square error.

    Args:
        y_true: observed values given in a list (or numpy array).
        y_pred: predicted values given in a list (or numpy array).

    Returns:
        mean absolute percent error.
    """
    return math.sqrt(mean_squared_error(y_true, y_pred))


@add_finite_filter_to_scorer
def correlation(y_true, y_pred):
    """Calculates correlation.

    Args:
        y_true: observed values given in a list (or numpy array).
        y_pred: predicted values given in a list (or numpy array).

    Returns:
        correlation.
    """
    if np.unique(y_true).size == 1:
        warnings.warn("y_true is constant. Correlation is not defined.")
        return None
    elif np.unique(y_pred).size == 1:
        warnings.warn("y_pred is constant. Correlation is not defined.")
        return None
    else:
        return pearsonr(y_true, y_pred)[0]


@add_finite_filter_to_scorer
def quantile_loss(y_true, y_pred, q: float = 0.95):
    """Calculates the quantile (pinball) loss with quantile q of y_true and y_pred.

    Args:
        y_true: one column of true values.  list or numpy array.
        y_pred: one column of predicted values.  list or numpy array.
        q: the quantile.  float.

    Returns:
        quantile loss.  float.
    """
    return np.where(y_true < y_pred, (1 - q) * (y_pred - y_true), q * (y_true - y_pred)).mean()


def quantile_loss_q(q):
    """Returns quantile loss function for the specified quantile"""

    def quantile_loss_wrapper(y_true, y_pred):
        return quantile_loss(y_true, y_pred, q=q)

    return quantile_loss_wrapper


def fraction_within_bands(observed, lower, upper):
    """Computes the fraction of observed values between lower and upper.

    Args:
        observed: pd.Series or np.array, numeric, observed values.
        lower: pd.Series or np.array, numeric, lower bound.
        upper: pd.Series or np.array, numeric, upper bound.

    Returns:
        float between 0.0 and 1.0.
    """
    observed, lower, upper = valid_elements_for_evaluation(
        reference_arrays=[observed],
        arrays=[lower, upper],
        reference_array_names="y_true",
        drop_leading_only=False,
        keep_inf=False,
    )
    lower, upper, observed = valid_elements_for_evaluation(
        reference_arrays=[lower, upper],
        arrays=[observed],
        reference_array_names="lower/upper bound",
        drop_leading_only=True,
        keep_inf=False,
    )
    num_reversed = np.count_nonzero(upper < lower)
    if num_reversed > 0:
        warnings.warn(f"{num_reversed} of {len(observed)} upper bound values are smaller than the lower bound")
    return np.count_nonzero((observed > lower) & (observed < upper)) / len(observed) if len(observed) > 0 else None


def fraction_outside_tolerance(y_true, y_pred, rtol=0.05):
    """Returns the fraction of predicted values whose relative difference
    from the true value is strictly greater than ``rtol``.

    Args:
        y_true: True values.  ``list`` or ``numpy.array``.
        y_pred: Predicted values.  ``list`` or ``numpy.array``.
        rtol: Relative error tolerance.
            For example, 0.05 allows for 5% relative error.  ``float``, default 0.05.

    Returns:
        frac_outside_tolerance: Fraction of values outside tolerance. (0.0 to 1.0).  float.
            A value is outside tolerance if `numpy.isclose`
            with the specified ``rtol`` and ``atol=0.0`` returns False.
    """
    y_true, y_pred = valid_elements_for_evaluation(
        reference_arrays=[y_true],
        arrays=[y_pred],
        reference_array_names="y_true",
        drop_leading_only=False,
        keep_inf=False,
    )
    y_pred, y_true = valid_elements_for_evaluation(
        reference_arrays=[y_pred],
        arrays=[y_true],
        reference_array_names="y_pred",
        drop_leading_only=True,
        keep_inf=True,
    )  # Keeps inf from prediction.
    return np.mean(~np.isclose(y_pred, y_true, rtol=rtol, atol=0.0))


def prediction_band_width(observed, lower, upper):
    """Computes the prediction band width, expressed as a % relative to observed.

    Width is defined as average ratio of (upper-lower)/observed.

    Args:
        observed: pd.Series or np.array, numeric, observed values.
        lower: pd.Series or np.array, numeric, lower bound.
        upper: pd.Series or np.array, numeric, upper bound.

    Returns:
        float, average percentage width.
    """
    observed, lower, upper = valid_elements_for_evaluation(
        reference_arrays=[observed],
        arrays=[lower, upper],
        reference_array_names="y_true",
        drop_leading_only=False,
        keep_inf=False,
    )
    lower, upper, observed = valid_elements_for_evaluation(
        reference_arrays=[lower, upper],
        arrays=[observed],
        reference_array_names="lower/upper bound",
        drop_leading_only=True,
        keep_inf=True,
    )  # Keeps inf from prediction.
    observed = np.abs(observed)
    num_reversed = np.count_nonzero(upper < lower)
    if num_reversed > 0:
        warnings.warn(f"{num_reversed} of {len(observed)} upper bound values are smaller than the lower bound")
    # if there are 0s in observed, relative size is undefined
    return 100.0 * np.mean(np.abs(upper - lower) / observed) if len(observed) > 0 and observed.min() > 0 else None


def mean_interval_score(observed, lower, upper, coverage):
    """Calculates the mean interval score.

    If an observed value falls within the interval, the score is simply the width of the interval.
    If an observed value falls outside the interval, the score is the width of the interval plus an error term
    proportional to distance between the actual and its closest interval boundary.
    The proportionality constant is 2.0 / (1.0 - `coverage`).
    See `Strictly Proper Scoring Rules, Prediction, and Estimation, Tilmann Gneiting and Adrian E. Raftery, 2007,
    Journal of the American Statistical Association, Volume 102, 2007 - Issue 477`.

    Args:
        observed: Numeric, observed values.  ``pandas.Series`` or ``numpy.array``.
        lower: Numeric, lower bound.  ``pandas.Series`` or ``numpy.array``.
        upper: Numeric, upper bound.  ``pandas.Series`` or ``numpy.array``.
        coverage: Intended coverage of the prediction bands (0.0 to 1.0).  ``float``.

    Returns:
        mean_interval_score: The mean interval score.  ``float``.
    """
    if not 0.0 <= coverage <= 1.0:
        raise ValueError(f"coverage must be in [0.0, 1.0]; got {coverage!r}.")
    observed, lower, upper = valid_elements_for_evaluation(
        reference_arrays=[observed],
        arrays=[lower, upper],
        reference_array_names="y_true",
        drop_leading_only=False,
        keep_inf=False,
    )
    lower, upper, observed = valid_elements_for_evaluation(
        reference_arrays=[lower, upper],
        arrays=[observed],
        reference_array_names="lower/upper bounds",
        drop_leading_only=False,
        keep_inf=True,
    )
    interval_width = upper - lower
    error_lower = np.where(observed < lower, 2.0 * (lower - observed) / (1.0 - coverage), 0.0)
    error_upper = np.where(observed > upper, 2.0 * (observed - upper) / (1.0 - coverage), 0.0)
    interval_score = interval_width + error_lower + error_upper
    return np.mean(interval_score)


def calc_pred_coverage(observed, predicted, lower, upper, coverage):
    """Calculates the prediction coverages: prediction band width, prediction band coverage etc.

    Args:
        observed: pd.Series or np.array, numeric, observed values.
        predicted: pd.Series or np.array, numeric, predicted values.
        lower: pd.Series or np.array, numeric, lower bound.
        upper: pd.Series or np.array, numeric, upper bound.
        coverage: float, intended coverage of the prediction bands (0.0 to 1.0).

    Returns:
        prediction band width, prediction band coverage etc.
    """
    observed, predicted, lower, upper = valid_elements_for_evaluation(
        reference_arrays=[observed],
        arrays=[predicted, lower, upper],
        reference_array_names="y_true",
        drop_leading_only=False,
        keep_inf=False,
    )
    predicted, lower, upper, observed = valid_elements_for_evaluation(
        reference_arrays=[predicted, lower, upper],
        arrays=[observed],
        reference_array_names="y_pred and lower/upper bounds",
        drop_leading_only=True,
        keep_inf=True,
    )
    metrics = {}

    if len(observed) > 0:
        # Relative size of prediction bands vs actual, as a percent.
        enum = ValidationMetricEnum.BAND_WIDTH
        metric_func = enum.get_metric_func()
        metrics.update({PREDICTION_BAND_WIDTH: metric_func(observed, lower, upper)})

        enum = ValidationMetricEnum.BAND_COVERAGE
        metric_func = enum.get_metric_func()
        # Fraction of observations within the bands.
        metrics.update({PREDICTION_BAND_COVERAGE: metric_func(observed, lower, upper)})
        # Fraction of observations within the lower band.
        metrics.update({LOWER_BAND_COVERAGE: metric_func(observed, lower, predicted)})
        # Fraction of observations within the upper band.
        metrics.update({UPPER_BAND_COVERAGE: metric_func(observed, predicted, upper)})
        # Difference between actual and intended coverage.
        metrics.update({COVERAGE_VS_INTENDED_DIFF: (metrics[PREDICTION_BAND_COVERAGE] - coverage)})

        # Mean interval score.
        enum = ValidationMetricEnum.MEAN_INTERVAL_SCORE
        metric_func = enum.get_metric_func()
        metric_name = enum.get_metric_name()
        metrics.update({metric_name: (metric_func(observed, lower, upper, coverage))})
    return metrics


def elementwise_residual(true_val, pred_val):
    """The residual between a single true and predicted value.

    Args:
        true_val: True value.  float.
        pred_val: Predicted value.  float.

    Returns:
        residual: The residual, true minus predicted.  float.
    """
    return true_val - pred_val


def elementwise_absolute_error(true_val, pred_val):
    """The absolute error between a single true and predicted value.

    Args:
        true_val: True value.  float.
        pred_val: Predicted value.  float.

    Returns:
        residual: Absolute error, |true_val - pred_val|.  float.
    """
    return abs(true_val - pred_val)


def elementwise_squared_error(true_val, pred_val):
    """The absolute error between a single true and predicted value.

    Args:
        true_val: True value.  float.
        pred_val: Predicted value.  float.

    Returns:
        residual: Squared error, (true_val - pred_val)^2.  float.
    """
    return (true_val - pred_val) ** 2


def elementwise_absolute_percent_error(true_val, pred_val):
    """The absolute percent error between a single true and predicted value.

    Args:
        true_val: True value.  float.
        pred_val: Predicted value.  float.

    Returns:
        percent_error: Percent error, pred_val / true_val - 1.  float.
    """
    if true_val == 0:
        warnings.warn("true_val is 0. Percent error is undefined.")
        return np.nan
    elif true_val < 1e-8:
        warnings.warn("true_val is less than 1e-8. Percent error is very likely highly volatile.")
    return 100 * abs(true_val - pred_val) / abs(true_val)


def elementwise_symmetric_absolute_percent_error(true_val, pred_val):
    """The symmetric absolute percent error between a single true and predicted value.

    Args:
        true_val: True value.  float.
        pred_val: Predicted value.  float.

    Returns:
        symmetric_absolute_percent_error: Symmetric Absolute Percent error,
            abs(true_val - pred_val) / (abs(true_val) + abs(pred_val)).  float.
    """
    denominator = abs(true_val) + abs(pred_val)
    if denominator == 0:
        warnings.warn("true_val and pred_val are 0. Symmetric absolute percent error is undefined.")
        return None
    elif denominator < 1e-8:
        warnings.warn("denominator contains very small values. Symmetric absolute percent error is " "very likely highly volatile.")
    return 100 * abs(true_val - pred_val) / (abs(true_val) + abs(pred_val))


def elementwise_quantile(true_val, pred_val, q):
    """The quantile loss between a single true and predicted value.

    Args:
        true_val: True value.  float.
        pred_val: Predicted value.  float.
        q: The quantile.

    Returns:
        quantile_loss: Quantile loss, absolute error weighed by ``q``
            for underpredictions and ``1-q`` for overpredictions.  float.
    """
    weight = 1 - q if true_val < pred_val else q
    return weight * abs(true_val - pred_val)


def elementwise_outside_tolerance(true_val, pred_val, rtol=0.05):
    """Whether the relative difference between ``pred_val`` and
    ``true_val`` is strictly greater than ``rtol``.

    Args:
        true_val: True value.  float.
        pred_val: Predicted value.  float.
        rtol: Relative error tolerance.
            For example, 0.05 allows for 5% relative error.  float, default 0.05.

    Returns:
        is_outside_tolerance: 1.0 if the error is outside tolerance, else 0.0.  float.
            A value is outside tolerance if `numpy.isclose`
            with the specified ``rtol`` and ``atol=0.0`` returns False.
    """
    return 0.0 if np.isclose(pred_val, true_val, rtol=rtol, atol=0.0) else 1.0


def elementwise_within_bands(true_val, lower_val, upper_val):
    """Whether ``true_val`` is strictly between ``lower_val`` and ``upper_val``.

    Args:
        true_val: True value.  float.
        lower_val: Lower bound.  float.
        upper_val: Upper bound.  float.

    Returns:
        is_within_band: 1.0 if error is strictly within the limits, else 0.0.  float.
    """
    return 1.0 if lower_val < true_val < upper_val else 0.0


class ElementwiseEvaluationMetricEnum(Enum):
    """Evaluation metrics for a single element.
    This is the function computed on each row, before aggregation across records.
    The aggregation function can be mean, median, quantile, max, sqrt of the mean, etc.

    For example, AbsoluteError followed by mean aggregation gives MeanAbsoluteError.
    EvaluationMetricEnum is more efficient if it can be used directly.
    """

    ElementwiseEvaluationMetric = namedtuple("ElementwiseEvaluationMetric", "func, name, args")
    """
    The elementwise evaluation function and associated metadata.

        ``"func"`` : `callable`
            Elementwise function.
        ``"name"`` : `str`
            Short name for the metric.
        ``"args"`` : `list` [`str`]
            Description of function arguments.
            e.g. [actual, predicted]
    """
    Residual = ElementwiseEvaluationMetric(elementwise_residual, "residual", [ACTUAL_COL, PREDICTED_COL])
    """Residual, true-pred"""
    AbsoluteError = ElementwiseEvaluationMetric(elementwise_absolute_error, "absolute_error", [ACTUAL_COL, PREDICTED_COL])
    """Absolute error, abs(true-pred)"""
    SquaredError = ElementwiseEvaluationMetric(elementwise_squared_error, "squared_error", [ACTUAL_COL, PREDICTED_COL])
    """Squared error, (true-pred)^2"""
    AbsolutePercentError = ElementwiseEvaluationMetric(elementwise_absolute_percent_error, "absolute_percent_error", [ACTUAL_COL, PREDICTED_COL])
    """Absolute percent error, abs(true-pred)/abs(true)"""
    SymmetricAbsolutePercentError = ElementwiseEvaluationMetric(
        elementwise_symmetric_absolute_percent_error,
        "symmetric_absolute_percent_error",
        [ACTUAL_COL, PREDICTED_COL],
    )
    """Symmetric absolute percent error, abs(true - pred)/ (abs(true) + abs(pred))"""
    Quantile80 = ElementwiseEvaluationMetric(partial(elementwise_quantile, q=0.80), "quantile_loss_80", [ACTUAL_COL, PREDICTED_COL])
    """Quantile loss with q=0.80"""
    Quantile90 = ElementwiseEvaluationMetric(partial(elementwise_quantile, q=0.90), "quantile_loss_90", [ACTUAL_COL, PREDICTED_COL])
    """Quantile loss with q=0.90"""
    Quantile95 = ElementwiseEvaluationMetric(partial(elementwise_quantile, q=0.95), "quantile_loss_95", [ACTUAL_COL, PREDICTED_COL])
    """Quantile loss with q=0.95"""
    Quantile99 = ElementwiseEvaluationMetric(partial(elementwise_quantile, q=0.99), "quantile_loss_99", [ACTUAL_COL, PREDICTED_COL])
    """Quantile loss with q=0.99"""
    OutsideTolerance5 = ElementwiseEvaluationMetric(
        partial(elementwise_outside_tolerance, rtol=0.05),
        "outside_tolerance_5p",
        [ACTUAL_COL, PREDICTED_COL],
    )
    """Whether the predicted value deviates more than 5% from the true value"""
    Coverage = ElementwiseEvaluationMetric(elementwise_within_bands, "coverage", [ACTUAL_COL, PREDICTED_LOWER_COL, PREDICTED_UPPER_COL])
    """Whether the actual value is within the prediction interval"""

    def get_metric_func(self):
        """Returns the metric function"""
        return self.value.func

    def get_metric_name(self):
        """Returns the short name"""
        return self.value.name

    def get_metric_args(self):
        """Returns the expected argument list"""
        return self.value.args


class EvaluationMetricEnum(Enum):
    """Valid evaluation metrics.
    The values tuple is ``(score_func: callable, greater_is_better: boolean, short_name: str)``

    ``add_finite_filter_to_scorer`` is added to the metrics that are directly imported from
    ``sklearn.metrics`` (e.g. ``mean_squared_error``) to ensure that the metric gets calculated
    even when inputs have missing values.
    """

    Correlation = (correlation, True, "CORR")
    """Pearson correlation coefficient between forecast and actuals. Higher is better."""
    CoefficientOfDetermination = (add_finite_filter_to_scorer(r2_score), True, "R2")
    """Coefficient of determination. See `sklearn.metrics.r2_score`. Higher is better. Equals `1.0 - mean_squared_error / variance(actuals)`."""
    MeanSquaredError = (add_finite_filter_to_scorer(mean_squared_error), False, "MSE")
    """Mean squared error, the average of squared differences,
    see `sklearn.metrics.mean_squared_error`."""
    RootMeanSquaredError = (root_mean_squared_error, False, "RMSE")
    """Root mean squared error, the square root of `sklearn.metrics.mean_squared_error`"""
    MeanAbsoluteError = (add_finite_filter_to_scorer(mean_absolute_error), False, "MAE")
    """Mean absolute error, average of absolute differences,
    see `sklearn.metrics.mean_absolute_error`."""
    MedianAbsoluteError = (add_finite_filter_to_scorer(median_absolute_error), False, "MedAE")
    """Median absolute error, median of absolute differences,
    see `sklearn.metrics.median_absolute_error`."""
    MeanAbsolutePercentError = (mean_absolute_percent_error, False, "MAPE")
    """Mean absolute percent error, error relative to actuals expressed as a %,
    see `wikipedia MAPE <https://en.wikipedia.org/wiki/Mean_absolute_percentage_error>`_."""
    MedianAbsolutePercentError = (median_absolute_percent_error, False, "MedAPE")
    """Median absolute percent error, median of error relative to actuals expressed as a %,
    a median version of the MeanAbsolutePercentError, less affected by extreme values."""
    SymmetricMeanAbsolutePercentError = (symmetric_mean_absolute_percent_error, False, "sMAPE")
    """Symmetric mean absolute percent error, error relative to (actuals+forecast) expressed as a %.
    Note that we do not include a factor of 2 in the denominator, so the range is 0% to 100%,
    see `wikipedia sMAPE <https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error>`_."""
    Quantile80 = (quantile_loss_q(0.80), False, "Q80")
    """Quantile loss with q=0.80::

        np.where(y_true < y_pred, (1 - q) * (y_pred - y_true), q * (y_true - y_pred)).mean()
    """
    Quantile95 = (quantile_loss_q(0.95), False, "Q95")
    """Quantile loss with q=0.95::

        np.where(y_true < y_pred, (1 - q) * (y_pred - y_true), q * (y_true - y_pred)).mean()
    """
    Quantile99 = (quantile_loss_q(0.99), False, "Q99")
    """Quantile loss with q=0.99::

        np.where(y_true < y_pred, (1 - q) * (y_pred - y_true), q * (y_true - y_pred)).mean()
    """
    FractionOutsideTolerance1 = (
        partial(fraction_outside_tolerance, rtol=0.01),
        False,
        "OutsideTolerance1p",
    )
    """Fraction of forecasted values that deviate more than 1% from the actual"""
    FractionOutsideTolerance2 = (
        partial(fraction_outside_tolerance, rtol=0.02),
        False,
        "OutsideTolerance2p",
    )
    """Fraction of forecasted values that deviate more than 2% from the actual"""
    FractionOutsideTolerance3 = (
        partial(fraction_outside_tolerance, rtol=0.03),
        False,
        "OutsideTolerance3p",
    )
    """Fraction of forecasted values that deviate more than 3% from the actual"""
    FractionOutsideTolerance4 = (
        partial(fraction_outside_tolerance, rtol=0.04),
        False,
        "OutsideTolerance4p",
    )
    """Fraction of forecasted values that deviate more than 4% from the actual"""
    FractionOutsideTolerance5 = (
        partial(fraction_outside_tolerance, rtol=0.05),
        False,
        "OutsideTolerance5p",
    )
    """Fraction of forecasted values that deviate more than 5% from the actual"""

    def get_metric_func(self):
        """Returns the metric function."""
        return self.value[0]

    def get_metric_greater_is_better(self):
        """Returns the greater_is_better boolean."""
        return self.value[1]

    def get_metric_name(self):
        """Returns the short name."""
        return self.value[2]


class ValidationMetricEnum(Enum):
    """Valid diagnostic metrics.
    The values tuple is ``(score_func: callable, greater_is_better: boolean, short_name: str)``
    """

    BAND_WIDTH = (prediction_band_width, False, "band_width")
    BAND_COVERAGE = (fraction_within_bands, True, "band_coverage")
    MEAN_INTERVAL_SCORE = (mean_interval_score, False, "MIS")
    """Mean interval score"""

    def get_metric_func(self):
        """Returns the metric function."""
        return self.value[0]

    def get_metric_greater_is_better(self):
        """Returns the greater_is_better boolean."""
        return self.value[1]

    def get_metric_name(self):
        """Returns the short name."""
        return self.value[2]
