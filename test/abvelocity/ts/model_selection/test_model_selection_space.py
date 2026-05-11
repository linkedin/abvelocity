# BSD 2-CLAUSE LICENSE
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# Original author: Reza Hosseini
"""Tests for SearchSpace and ParamGroup."""

import pytest
from abvelocity.ts.model_selection.space import ParamGroup, SearchSpace


def test_param_group_candidates_cartesian():
    group = ParamGroup(name="trend", params={"reg": [0.4, 0.6], "yearly": [4, 6, 8]})
    cands = group.candidates()
    assert len(cands) == 6
    # Every reg paired with every yearly.
    pairs = {(c["reg"], c["yearly"]) for c in cands}
    assert pairs == {(0.4, 4), (0.4, 6), (0.4, 8), (0.6, 4), (0.6, 6), (0.6, 8)}


def test_param_group_validation_empty_dict():
    with pytest.raises(ValueError, match="non-empty dict"):
        ParamGroup(name="x", params={})


def test_param_group_validation_empty_value_list():
    with pytest.raises(ValueError, match="non-empty list"):
        ParamGroup(name="x", params={"a": []})


def test_param_group_validation_empty_name():
    with pytest.raises(ValueError, match="non-empty string"):
        ParamGroup(name="", params={"a": [1]})


def test_search_space_flat_constructor():
    space = SearchSpace.flat({"a": [1, 2], "b": ["x", "y"]})
    assert len(space.groups) == 1
    assert space.groups[0].name == "all"
    cands = space.cartesian_candidates()
    assert len(cands) == 4
    assert {(c["a"], c["b"]) for c in cands} == {(1, "x"), (1, "y"), (2, "x"), (2, "y")}


def test_search_space_requires_at_least_one_group():
    with pytest.raises(ValueError, match="at least one"):
        SearchSpace(groups=[])


def test_search_space_unique_group_names():
    g1 = ParamGroup(name="dup", params={"a": [1]})
    g2 = ParamGroup(name="dup", params={"b": [2]})
    with pytest.raises(ValueError, match="unique"):
        SearchSpace(groups=[g1, g2])


def test_search_space_all_param_names_preserves_order():
    space = SearchSpace(
        groups=[
            ParamGroup(name="g1", params={"reg": [0.5], "yearly": [4]}),
            ParamGroup(name="g2", params={"ridge": [5]}),
        ]
    )
    assert space.all_param_names() == ["reg", "yearly", "ridge"]


def test_search_space_cartesian_across_groups():
    space = SearchSpace(
        groups=[
            ParamGroup(name="g1", params={"a": [1, 2]}),
            ParamGroup(name="g2", params={"b": [10, 20]}),
        ]
    )
    cands = space.cartesian_candidates()
    assert len(cands) == 4
    pairs = {(c["a"], c["b"]) for c in cands}
    assert pairs == {(1, 10), (1, 20), (2, 10), (2, 20)}


def test_search_space_cartesian_conflicting_param_raises():
    space = SearchSpace(
        groups=[
            ParamGroup(name="g1", params={"a": [1, 2]}),
            ParamGroup(name="g2", params={"a": [3, 4]}),
        ]
    )
    with pytest.raises(ValueError, match="conflicting values"):
        space.cartesian_candidates()


def test_stage_candidates_freezes_prior_winners():
    space = SearchSpace(
        groups=[
            ParamGroup(name="trend", params={"reg": [0.4, 0.6]}),
            ParamGroup(name="seas", params={"yearly": [4, 8]}),
        ]
    )
    frozen = {"reg": 0.6}
    pairs = list(space.stage_candidates(stage_index=1, frozen=frozen))
    assert len(pairs) == 2
    stage_only_dicts = [stage_only for stage_only, _ in pairs]
    full_dicts = [full for _, full in pairs]
    assert {tuple(d.items()) for d in stage_only_dicts} == {(("yearly", 4),), (("yearly", 8),)}
    # All full dicts carry the frozen reg.
    assert all(d["reg"] == 0.6 for d in full_dicts)


def test_stage_candidates_reopen_re_sweeps_earlier_param():
    space = SearchSpace(
        groups=[
            ParamGroup(name="trend", params={"reg": [0.4, 0.6]}),
            ParamGroup(
                name="seas",
                params={"yearly": [4, 8]},
                reopen=["reg"],  # re-sweep reg alongside yearly
            ),
        ]
    )
    pairs = list(space.stage_candidates(stage_index=1, frozen={"reg": 0.4}))
    # 2 reg × 2 yearly = 4 candidates.
    assert len(pairs) == 4
    stage_only_keys = {tuple(sorted(d.keys())) for d, _ in pairs}
    assert stage_only_keys == {("reg", "yearly")}


def test_stage_candidates_reopen_unknown_param_raises():
    space = SearchSpace(
        groups=[
            ParamGroup(name="g1", params={"a": [1]}),
            ParamGroup(name="g2", params={"b": [2]}, reopen=["nope"]),
        ]
    )
    with pytest.raises(ValueError, match="reopen"):
        list(space.stage_candidates(stage_index=1, frozen={"a": 1}))


def test_param_group_to_dict_round_trips_via_json():
    import json
    g = ParamGroup(name="regression", params={"fit_algorithm": ["ridge", "linear"]}, reopen=None)
    d = g.to_dict()
    assert d == {"name": "regression", "params": {"fit_algorithm": ["ridge", "linear"]}, "augment": False, "reopen": None, "size": 2}
    # JSON-serializable.
    assert json.dumps(d)


def test_search_space_to_dict_includes_cartesian_size():
    space = SearchSpace(
        groups=[
            ParamGroup(name="reg", params={"fit_algorithm": ["ridge", "linear"]}),
            ParamGroup(name="cp", params={"changepoint_reg": [0.01, 1.0]}),
        ]
    )
    d = space.to_dict()
    assert d["cartesian_size"] == 4
    assert [g["name"] for g in d["groups"]] == ["reg", "cp"]
    assert d["groups"][0]["size"] == 2


def test_search_space_str_lists_each_stage_with_params():
    space = SearchSpace(
        groups=[
            ParamGroup(name="reg", params={"fit_algorithm": ["ridge", "linear"]}),
            ParamGroup(name="cp", params={"changepoint_reg": [0.01, 1.0]}, reopen=["fit_algorithm"]),
        ]
    )
    text = str(space)
    assert "SearchSpace(groups=2, cartesian=" in text
    assert "[stage 1]" in text
    assert "[stage 2]" in text
    assert "fit_algorithm: ['ridge', 'linear']" in text
    assert "reopen=['fit_algorithm']" in text
