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

from pathlib import Path

from abvelocity.param.get_metric_conf import (
    get_hocon_conf,
    get_metric_conf,
    get_metric_names_via_conf,
)


def test_get_hocon_conf():
    path = Path(__file__).parents[1].joinpath("test_data/hocon")
    file = "hocon_example.conf"

    config = get_hocon_conf(file=file, path=path)

    # Get the first element from the 'mapreduce' list
    first_mapreduce = config.get_list("mapreduce")[0]

    # Assert the nested value for '"override.mapreduce.map.memory.mb"' (with quotes)
    assert first_mapreduce['"override.mapreduce.map.memory.mb"'] == 5120


def test_get_metric_conf():
    config = get_metric_conf(metric_family_name="RandomProduct_sessions")

    assert config["datafiles"][0]["metrics"][0]["name"] == "n_wvmp_page_views"


def test_get_metric_names_via_conf():
    metric_names = get_metric_names_via_conf(metric_family_name="RandomProduct_sessions")

    assert metric_names[0] == "n_wvmp_page_views"
