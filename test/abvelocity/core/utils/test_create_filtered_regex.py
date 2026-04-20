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

from abvelocity.core.utils.create_filtered_regex import create_filtered_regex


def test_pure_exclusion_animals():
    """Test that exclusions correctly filter out animal strings."""
    exclusions = ["_horse_", "dog"]
    pattern = create_filtered_regex(exclusions=exclusions)

    # Passes: no horse or dog
    assert re.match(pattern, "blue_bird") is not None
    # Fails: contains horse
    assert re.match(pattern, "fast_horse_run") is None
    # Fails: contains dog
    assert re.match(pattern, "lazy_dog_sleep") is None


def test_inclusion_overrules_exclusion_animals():
    """Test that an inclusion allows a string even if an exclusion is present."""
    # We want to allow a specific horse, but exclude all other horses and cats
    inclusions = ["golden_horse"]
    exclusions = ["_horse_", "cat"]
    pattern = create_filtered_regex(inclusions=inclusions, exclusions=exclusions)

    # PASS: Inclusion 'golden_horse' overrules the '_horse_' exclusion
    assert re.match(pattern, "the_golden_horse_wins") is not None

    # FAIL: Standard horse is excluded
    assert re.match(pattern, "standard_horse_race") is None

    # FAIL: Cat is excluded
    assert re.match(pattern, "black_cat_climb") is None

    # PASS: Neither hit
    assert re.match(pattern, "wild_wolf") is not None


def test_multiple_exclusions():
    """Verify that multiple animal exclusions work in combination."""
    exclusions = ["cat", "dog", "bird"]
    pattern = create_filtered_regex(exclusions=exclusions)

    assert re.match(pattern, "mountain_lion") is not None
    assert re.match(pattern, "small_cat") is None
    assert re.match(pattern, "big_dog") is None
    assert re.match(pattern, "blue_bird") is None


def test_empty_filters():
    """Ensure empty lists allow any animal string."""
    pattern = create_filtered_regex(inclusions=[], exclusions=[])
    assert re.match(pattern, "any_animal") is not None
    assert re.match(pattern, "cat_dog_horse") is not None


def test_special_characters_animals():
    """Verify that animals with special characters in names are escaped."""
    # Testing names that might contain regex-sensitive characters
    exclusions = ["cat+", "dog.1"]
    pattern = create_filtered_regex(exclusions=exclusions)

    # Should fail on literal 'cat+'
    assert re.match(pattern, "my_cat+_is_cool") is None
    # Should pass on 'catplus' because '+' was escaped to be a literal
    assert re.match(pattern, "my_catplus_is_cool") is not None
