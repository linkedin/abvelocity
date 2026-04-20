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
# ON ANY THEORY OF LAW, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# Original author: Reza Hosseini

from dataclasses import dataclass

# These are the deduping methods
UNDEDUPED = "undeduped"
CONSECUTIVE_DEDUPED = "consecutive_deduped"
FULLY_DEDUPED = "fully_deduped"
MAP = "map"

DEDUPING_MTHODS = [UNDEDUPED, CONSECUTIVE_DEDUPED, FULLY_DEDUPED, MAP]


@dataclass
class SeqInfo:
    deduping_method: str = ""
    output_table_name: str = ""
    max_seq_index: int = 5
    """The max seq element number used in sunburst plots"""


# Now we define a useful constant to cover all sequence types we may need.
SEQ_INFO_LIST = []

for method in DEDUPING_MTHODS:
    SEQ_INFO_LIST.append(SeqInfo(deduping_method=method))
