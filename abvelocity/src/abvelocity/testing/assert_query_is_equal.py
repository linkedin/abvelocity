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


def assert_query_is_equal(query1: str, query2: str) -> bool:
    """Compares two queries, ignoring whitespace, newlines, and case for keywords,
    but preserving case for string literals.
    """

    def standardize_query(query: str) -> str:
        # 1. Remove comments
        query = re.sub(r"--.*$", "", query, flags=re.MULTILINE)
        query = re.sub(r"/\*.*?\*/", "", query, flags=re.DOTALL)

        # 2. Replace all newlines with spaces
        query = query.replace("\n", " ")

        # 3. Normalize whitespace and convert to single spaces
        query = re.sub(r"\s+", " ", query).strip()

        # 4. Convert keywords to uppercase (after whitespace normalization)
        def replace_keyword(match):
            keyword = match.group(0)
            if keyword.startswith("'") or keyword.startswith('"'):
                return keyword  # String literal, keep as is
            return keyword.upper()

        query = re.sub(r"('[^']*'|\"[^\"]*\"|[a-zA-Z_]+)", replace_keyword, query)
        return query

    standardized_query1 = standardize_query(query1)
    standardized_query2 = standardize_query(query2)

    if standardized_query1 != standardized_query2:
        raise AssertionError(
            f"\n***\nquery1: {standardized_query1}\n is not same as \nquery2: {standardized_query2}"
        )

    return True
