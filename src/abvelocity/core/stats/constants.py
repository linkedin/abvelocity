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


# Map aggregation strings to pandas functions
AGG_MAP = {
    "SUM": lambda x: x.sum(),
    "MEAN": lambda x: x.mean(),
    "AVG": lambda x: x.mean(),  # Same as MEAN but included as its SQL standard.
    "COUNT": lambda x: x.count(),
    "MAX": lambda x: x.max(),
    "MIN": lambda x: x.min(),
}
"""
Mapping of aggregation strings to corresponding pandas functions.

Defines the supported aggregation operations for numerator and denominator calculations.
Keys are aggregation names (e.g., 'SUM', 'MEAN'), and values are lambda functions that
apply the corresponding pandas Series method.
"""

STANDARD_VALUES_NUMER = ["sum_numer", "mean_numer"]
"""
List of standard value names for metrics without a denominator.

Used when compute_standard_values is True and no denominator is present in the Metric.
Contains keys for sum and mean of the numerator.
"""

STANDARD_VALUES_RATIO = [
    "sum_numer",
    "mean_numer",
    "sum_denom",
    "mean_denom",
    "sum_ratio",
    "mean_ratio",
]
"""
List of standard value names for metrics with a denominator.

Used when compute_standard_values is True and a denominator is present in the Metric.
Contains keys for sum and mean of numerator, sum and mean of denominator, and ratio
of sums and means (sum(numerator)/sum(denominator), mean(numerator)/mean(denominator)).
"""
