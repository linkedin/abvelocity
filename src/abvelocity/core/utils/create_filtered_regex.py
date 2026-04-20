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

import re


def create_filtered_regex(inclusions: list[str] = None, exclusions: list[str] = None) -> str:
    """
    Creates a regex where inclusions overrule exclusions.

    Logic:
    1. If a string matches any item (substring) in 'inclusions', it is a valid match.
    2. If it doesn't match an inclusion, it must NOT contain any items (substrings) in 'exclusions'.
    """
    inclusions = inclusions or []
    exclusions = exclusions or []

    # Escape items to handle special characters (like . or +)
    inc_pattern = "|".join([re.escape(i) for i in inclusions])
    exc_lookaheads = "".join([f"(?!.*{re.escape(e)})" for e in exclusions])

    # Case 1: No filters provided
    if not inclusions and not exclusions:
        return r".*"

    # Case 2: Only exclusions provided
    if not inclusions:
        return rf"^{exc_lookaheads}.*$"

    # Case 3: Only inclusions provided
    if not exclusions:
        return rf".*({inc_pattern}).*"

    # Case 4: Both provided.
    # Logic: (Must have inclusion) OR (Must pass all exclusion lookaheads)
    return rf"(.*({inc_pattern}).*)|(^{exc_lookaheads}.*$)"
