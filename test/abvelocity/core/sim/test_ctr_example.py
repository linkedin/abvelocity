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

import pandas as pd
from abvelocity.core.sim.ctr_example import sim_ctr_data
from abvelocity.core.stats.param import StrataInfo

# Define the default population size used in sim_ctr_data for the test check
DEFAULT_POPULATION_SIZE = 20000


def test_sim_ctr_data():
    """
    Tests the sim_ctr_data function to ensure it returns the correct data
    types and that the calculated true values are reasonable (non-zero/expected keys).
    """

    # 1. Run the data simulation function
    df, strata_info, true_values = sim_ctr_data(
        population_size=DEFAULT_POPULATION_SIZE,
        seed=42,  # Use a specific seed for predictable test results
    )

    # 2. Check the output structure and types

    # Assert that the main data is a pandas DataFrame
    assert isinstance(df, pd.DataFrame)
    # Assert that the DataFrame has the expected number of records
    assert len(df) == DEFAULT_POPULATION_SIZE

    # Assert that the strata info is the expected type
    assert isinstance(strata_info, StrataInfo)

    # Assert that the true values are returned as a dictionary
    assert isinstance(true_values, dict)

    # 3. Check the content of the true_values dictionary
    expected_keys = ["impressions_diff", "impressions_pct_diff", "ctr_diff", "ctr_pct_diff"]
    # Assert all expected metric keys are present
    assert all(key in true_values for key in expected_keys)

    # 4. Check for reasonable values (e.g., non-zero, expected direction)

    # Based on the impacts defined in the module, the impressions delta should be positive.
    # The expected true difference is around 13.939
    assert true_values["impressions_diff"] > 10.0

    # The expected true CTR difference should also be positive.
    # The expected true percentage difference is around 6.5%
    assert true_values["ctr_pct_diff"] > 5.0
