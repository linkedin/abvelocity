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

import re


def create_name_from_regex(regex_str: str) -> str:
    """
    Parses a regex string to create a human-readable name based on included and excluded keywords.

    Args:
        regex_str (str): The regex string to be parsed.

    Returns:
        str: A sanitized, human-readable string representing the regex filters.

    Examples:
        >>> regex = r"^(?!.*suspen)(?!.*cancel)(?!.*_ios_)(?!.*_android_).*(?:_free|survival).*$"
        >>> create_name_from_regex(regex)
        'no_suspen_no_cancel_no_ios_no_android_with_free_with_survival'

        >>> regex = r".*(?:_ios_|_android_).*$"
        >>> create_name_from_regex(regex)
        'with_ios_with_android'
    """
    # Find all negative lookaheads and extract the keywords
    negative_pattern = r"\(\?!.\*([a-zA-Z0-9_]+)\)"
    negative_matches = re.findall(negative_pattern, regex_str)

    # Prepend "no_" to all negative match keywords
    negative_names = [f"no_{name}" for name in negative_matches]

    # Find the positive-match non-capturing group and extract keywords
    positive_pattern = r"\(\?:\*?([a-zA-Z0-9_\|]+)\*?\)"
    positive_match = re.search(positive_pattern, regex_str)

    positive_names = []
    if positive_match:
        keywords = positive_match.group(1).split("|")
        # Prepend "w_" to all positive match keywords
        positive_names = [f"w_{keyword.strip()}" for keyword in keywords if keyword.strip()]

    # Combine the lists and join into a final string, ensuring no empty elements
    all_names = negative_names + positive_names
    cleaned_names = [name for name in all_names if name]

    result = "_".join(cleaned_names)

    # Replace double underscores with single underscores
    result = result.replace("__", "_")

    # Remove trailing underscores
    return result.rstrip("_")
