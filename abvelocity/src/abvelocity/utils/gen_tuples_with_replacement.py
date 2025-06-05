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
# author: Reza Hosseini

from itertools import product


def gen_tuples_with_replacement(original_tuple: tuple, replacement: any) -> list[tuple]:
    """
    Generate all possible tuples by replacing any element of the original tuple
    with a constant replacement value, including the original tuple but excluding the tuple where all elements are the replacement value.

    Args:
        original_tuple: The original tuple of elements.
        replacement: The value to replace elements with.

    Returns:
        list[tuple]: A list of tuples with each possible combination of elements replaced by the replacement value.
    """
    n = len(original_tuple)
    # Generate all combinations of True/False of length n
    all_combinations = list(product([True, False], repeat=n))

    generated_tuples = []

    for combination in all_combinations:
        # Create a new tuple based on the combination
        new_tuple = tuple(replacement if combination[i] else original_tuple[i] for i in range(n))
        # Add the new tuple to the generated_tuples list
        generated_tuples.append(new_tuple)

    # Remove the tuple where all elements are the replacement value
    all_replacement_tuple = (replacement,) * n
    if all_replacement_tuple in generated_tuples:
        generated_tuples.remove(all_replacement_tuple)

    # Assert the size of the generated_tuples list
    expected_size = (2**n) - 1
    assert (
        len(generated_tuples) == expected_size
    ), f"Expected size {expected_size}, but got {len(generated_tuples)}"

    return generated_tuples
