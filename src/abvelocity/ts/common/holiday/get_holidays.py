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
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""Holiday lookup utilities backed by the `holidays` package (vacanza/holidays).

This module replaces the former `holidays_ext` PyPI dependency, which was unmaintained
and had hard-coded date limits (e.g. Indonesia Nyepi only through 2022).  The `holidays`
package (>=0.27) natively covers all countries that `holidays_ext` added, with
algorithmically-computed dates that remain accurate indefinitely.

Public API is intentionally identical to `holidays_ext.get_holidays` so call-sites
require only an import change.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime
from typing import List, Optional

import holidays
import pandas as pd
from holidays.registry import COUNTRIES

# All valid country identifiers (ISO-2, ISO-3, and full class names like "UnitedStates").
# holidays>=0.27 uses lazy loaders so inspect.isclass() does not work; use the registry instead.
ALL_COUNTRY_NAMES: frozenset = frozenset(name for names in COUNTRIES.values() for name in names)

# ISO codes and class names that resolve to the United States.
# Halloween is added here because it is not a US federal holiday but has historically
# been included in the holiday feature set used by this package.
US_IDENTIFIERS = {"US", "USA", "UnitedStates"}

# ── Name sanitisation ─────────────────────────────────────────────────────────

PAREN_RE = re.compile(r"\s*\([^)]*\)")
MULTI_SPACE = re.compile(r"\s{2,}")

# Trailing " Day" is stripped from all names EXCEPT these stems, where "Day"
# is load-bearing for meaning or disambiguation.
KEEP_DAY_SUFFIX: frozenset = frozenset({"Boxing", "May", "National"})

# Cross-country canonical name map.
# Applied *after* sanitize_name (keys use the already-sanitised + " Day"-stripped form).
# Two purposes:
#   (A) unify holidays that represent the same real-world event across countries
#   (B) replace awkward 4-word truncations with a readable short name
#
#   key (sanitised + Day-stripped)          canonical        reason
#   ---------------------------------------------------------------
#   Universal Fraternization  →  New Years        (BR Jan 1 = New Year)
#   Workers                   →  Labor            (BR May 1)
#   Labour and Solidarity     →  Labor            (TR May 1)
#   Labour                    →  Labor            (spelling)
#   International Labor       →  Labor            (ID, NG May 1)
#   Chinese New Year          →  Lunar New Year   (CN; same festival as KR/ID)
#   Korean New Year           →  Lunar New Year   (KR Seollal)
#   Second Day of Christmas   →  Boxing Day       (DE, NL Dec 26)
#   Saint Stephens            →  Boxing Day       (IT Dec 26)
#   Day of Goodwill           →  Boxing Day       (ZA Dec 26)
#   Eid al-Adha Holiday       →  Eid al-Adha      (ID, NG multi-day naming)
#   Eid al-Fitr Holiday       →  Eid al-Fitr      (ID, NG multi-day naming)
#   Eid al-Fitr Second        →  Eid al-Fitr      (ID second day)
#   Assumption Of Mary        →  Assumption       (IT vs FR/ES)
#   Martin Luther King Jr     →  MLK              (period + truncation)
#   Commemoration of Ataturk Youth → Ataturk      (7-word TR holiday)
#   Democracy and National Unity   → National Unity    (TR Jul 15)
#   National Sovereignty and Childrens → National Sovereignty  (TR Apr 23)
#   National Day of Zumbi     →  Black Awareness  (BR Nov 20)
#
CANONICAL_NAMES: dict = {
    # (A) Cross-country equivalences — same real-world event, different local names
    #
    # Jan 1 — Brazil calls New Year "Universal Fraternization Day"
    "Universal Fraternization": "New Years",  # Brazil Jan 1
    #
    # May 1 — International Workers' Day
    "Workers": "Labor",  # Brazil May 1
    "Labour and Solidarity": "Labor",  # Turkey May 1
    "Labour": "Labor",  # spelling normalisation (GB etc.)
    "International Labor": "Labor",  # Indonesia, Nigeria etc.
    #
    # Lunar New Year — same festival across East/SE Asia
    "Chinese New Year": "Lunar New Year",  # China
    "Korean New Year": "Lunar New Year",  # Korea (Seollal)
    #
    # Dec 26 — all local names for the day after Christmas
    "Second Day of Christmas": "Boxing Day",  # Germany, Netherlands
    "Saint Stephens": "Boxing Day",  # Italy (Santo Stefano)
    "Day of Goodwill": "Boxing Day",  # South Africa
    #
    # Islamic holidays — multi-day / regional naming variants
    "Eid al-Adha Holiday": "Eid al-Adha",  # Indonesia, Nigeria extended holiday
    "Eid al-Fitr Holiday": "Eid al-Fitr",  # Indonesia, Nigeria extended holiday
    "Eid al-Fitr Second": "Eid al-Fitr",  # Indonesia second day
    #
    # Aug 15 — Assumption (same Catholic feast, different local phrasing)
    "Assumption Of Mary": "Assumption",  # Italy vs France/Spain
    #
    # (B) Long-name abbreviations (after 4-word truncation + Day strip)
    "Martin Luther King Jr": "MLK",
    "Commemoration of Ataturk Youth": "Ataturk",  # May 19 Turkey
    "Democracy and National Unity": "National Unity",  # Jul 15 Turkey
    "National Sovereignty and Childrens": "National Sovereignty",  # Apr 23 Turkey
    "National Day of Zumbi": "Black Awareness",  # Nov 20 Brazil
}


def sanitize_name(name: str, max_words: int = 4) -> str:
    """Normalise a holiday name for use as a model feature label.

    Rules applied in order:

    1. **Same-date collisions** — the library joins multiple holidays on one date
       with ``"; "``, e.g. ``"Dussehra; Gandhi Jayanti"``.  Keep the first entry.
    2. **Accent stripping** — ``"Atatürk"`` → ``"Ataturk"``.
    3. **Parenthetical removal** — ``"(estimated)"`` etc. stripped.
       ``"(observed)"`` / ``"(Observed)"`` is stripped here and appended as the
       ``"_observed"`` suffix after all other processing, so that observed holidays
       remain distinct from their canonical counterparts.
    4. **Punctuation** — periods, apostrophes, commas, colons, semicolons removed.
    5. **Whitespace collapse**.
    6. **Word limit** — keep at most *max_words* words (default 4).
    7. **Trailing " Day" strip** — e.g. ``"Christmas"`` → ``"Christmas"``,
       ``"Independence"`` → ``"Independence"``.  Exceptions: names whose stem
       is in ``KEEP_DAY_SUFFIX`` (``"Boxing"``, ``"May"``, ``"National"``) keep the
       suffix so ``"Boxing Day"`` / ``"May Day"`` / ``"National Day"`` are unchanged.
    """
    # (1) Multi-holiday entries
    name = name.split(";")[0].strip()
    # (2) Accent normalisation
    name = unicodedata.normalize("NFD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    # (3a) Detect and strip the "(observed)" marker — will be re-appended after step 7.
    _is_observed = bool(re.search(r"\([Oo]bserved\)", name))
    if _is_observed:
        name = re.sub(r"\s*\([Oo]bserved\)", "", name).strip()
    # (3b) Remove all remaining parenthetical content.
    name = PAREN_RE.sub("", name)
    # (4) Punctuation
    name = name.replace(".", "").replace(",", "").replace("'", "")
    name = re.sub(r"[;:]", "", name)
    # (5) Whitespace
    name = MULTI_SPACE.sub(" ", name).strip()
    # (6) Word limit
    words = name.split()
    if max_words and len(words) > max_words:
        name = " ".join(words[:max_words])
    # (7) Trailing " Day" strip
    if name.endswith(" Day"):
        stem = name[:-4]
        if stem not in KEEP_DAY_SUFFIX:
            name = stem
    # Re-append observed suffix so "New Year's Day (observed)" → "New Years_observed".
    if _is_observed:
        name = name + "_observed"
    return name


def normalize_name(name: str) -> str:
    """Sanitise a holiday name then apply the cross-country canonical mapping.

    Observed-holiday variants (``_observed`` suffix) are canonicalised on their
    base name so that, e.g., ``"Labour Day (observed)"`` → ``"Labor_observed"``.
    """
    sanitised = sanitize_name(name)
    observed_suffix = "_observed"
    if sanitised.endswith(observed_suffix):
        base = sanitised[: -len(observed_suffix)]
        return CANONICAL_NAMES.get(base, base) + observed_suffix
    return CANONICAL_NAMES.get(sanitised, sanitised)


def get_country_holidays(country: str, years: List[int], categories=None):
    """Return a HolidayBase dict for *country* across *years*.

    Accepts both ISO-3166 codes (``"US"``, ``"GB"``) and full class names
    (``"UnitedStates"``, ``"UnitedKingdom"``).  Always requests English names
    (``language='en_US'``) so holiday strings are consistent regardless of the
    country's native language default.

    Args:
        country: Country identifier.  ``str``.
        years: Years to include.  ``list`` of ``int``.
        categories: Holiday categories to include, optional.  Defaults to
            ``(holidays.PUBLIC, holidays.OPTIONAL)`` so that widely-observed
            holidays (e.g. Holi in India, Carnival in Brazil) are included even
            when they are not legally mandated public holidays.  Pass
            ``(holidays.PUBLIC,)`` to restrict to official public holidays only.
            ``tuple`` or None.

    Raises:
        AttributeError: For unknown country identifiers.
    """
    if categories is None:
        categories = (holidays.PUBLIC, holidays.OPTIONAL)

    # Validate via the registry (getattr returns a lazy loader, not a class, in holidays>=0.27).
    if country not in ALL_COUNTRY_NAMES:
        raise AttributeError(f"Holidays in {country} are not currently supported!")

    loader = getattr(holidays, country)
    try:
        obj = loader(years=years, language="en_US", categories=categories)
    except (KeyError, NotImplementedError):
        # Country does not support en_US translation; fall back to default language.
        try:
            obj = loader(years=years, categories=categories)
        except ValueError:
            # Country does not support all requested categories; fall back to PUBLIC only.
            obj = loader(years=years)
    except ValueError:
        # Country does not support all requested categories (e.g. no OPTIONAL for US);
        # fall back gracefully to PUBLIC only.
        try:
            obj = loader(years=years, language="en_US", categories=(holidays.PUBLIC,))
        except (KeyError, NotImplementedError):
            obj = loader(years=years, categories=(holidays.PUBLIC,))

    # Halloween: not a US public holiday but included in the holiday feature set used by this package.
    if country in US_IDENTIFIERS:
        for year in years:
            obj[date(year, 10, 31)] = "Halloween"

    # Normalise all names: strip special chars and unify cross-country equivalents.
    # Returns a plain dict so callers are not coupled to HolidayBase internals.
    return {d: normalize_name(raw_name) for d, raw_name in obj.items()}


def get_holiday(country_list: List[str], years: List[int], categories=None) -> dict:
    """Look up holidays for a list of countries and years.

    Args:
        country_list: Country identifiers — ISO-3166 codes (``"US"``) or full
            class names (``"UnitedStates"``).  ``list`` of ``str``.
        years: Years to include.  ``list`` of ``int``.
        categories: Holiday categories to include, optional.  Defaults to
            ``(holidays.PUBLIC, holidays.OPTIONAL)`` so that widely-observed
            holidays (e.g. Holi in India, Carnival in Brazil) are included even
            when they are not legally mandated public holidays.  Pass
            ``(holidays.PUBLIC,)`` to restrict to official public holidays only.
            ``tuple`` or None.

    Returns:
        ``dict`` mapping ``{country: HolidayBase}`` where each value is a
        date→name mapping.

    Raises:
        AttributeError: If a country identifier is not recognised.
    """
    return {country: get_country_holidays(country, years, categories=categories) for country in country_list}


def get_holiday_df(
    country_list: List[str],
    years: List[int],
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Generate a DataFrame with holidays for the given countries and years.

    Args:
        country_list: Country identifiers.  ``list`` of ``str``.
        years: Years to include.  ``list`` of ``int``.
        start_date: Optional inclusive lower bound — drops rows
            before this date if set.  ``pandas.Timestamp`` or None.
        end_date: Optional inclusive upper bound — drops rows after
            this date if set.  ``pandas.Timestamp`` or None.

    Returns:
        ``pandas.DataFrame`` with columns ``"ts"``, ``"country"``, ``"holiday"``,
        ``"country_holiday"``.
    """
    country_holidays = get_holiday(country_list=country_list, years=years)
    dfs = []
    for country, hols in country_holidays.items():
        temp_df = pd.DataFrame(list(hols.items()), columns=["ts", "holiday"])
        temp_df["ts"] = pd.to_datetime(temp_df["ts"])
        temp_df["country"] = country
        temp_df["country_holiday"] = temp_df["country"] + "_" + temp_df["holiday"]
        dfs.append(temp_df)
    df = pd.concat(dfs, axis=0).reset_index(drop=True)[["ts", "country", "holiday", "country_holiday"]]
    if start_date is not None:
        df = df[df["ts"] >= start_date]
    if end_date is not None:
        df = df[df["ts"] <= end_date]
    return df.reset_index(drop=True)


def get_available_holiday_lookup_countries(countries: Optional[List[str]] = None) -> List[str]:
    """Return country identifiers supported for holiday lookup.

    Args:
        countries: If provided, return only those from this list that are
            supported.  ``list`` of ``str``, optional.

    Returns:
        Sorted ``list`` of ``str`` — valid country identifiers (ISO codes,
        ISO-3 codes, and full class names such as ``"UnitedStates"``).
    """
    if countries is None:
        return sorted(ALL_COUNTRY_NAMES)
    return sorted([c for c in countries if c in ALL_COUNTRY_NAMES])


def get_available_holidays_in_countries(countries: List[str], year_start: Optional[int] = None, year_end: Optional[int] = None) -> dict:
    """Return all distinct holiday names per country over a year range.

    Args:
        countries: Country identifiers.  ``list`` of ``str``.
        year_start: First year (inclusive). Defaults to 1985.  ``int``,
            optional.
        year_end: Last year (inclusive). Defaults to the current year.  ``int``,
            optional.

    Returns:
        ``dict`` mapping ``{country: sorted list of holiday names}``.
    """
    if year_start is None:
        year_start = 1985
    if year_end is None:
        year_end = max(datetime.now().year, year_start)
    years = list(range(year_start, year_end + 1))
    country_holidays = get_holiday(country_list=countries, years=years)
    return {country: sorted(set(hols.values())) for country, hols in country_holidays.items()}


def get_available_holidays_across_countries(countries: List[str], year_start: Optional[int] = None, year_end: Optional[int] = None) -> List[str]:
    """Return all holiday names that appear in any of the given countries.

    Args:
        countries: Country identifiers.  ``list`` of ``str``.
        year_start: First year (inclusive). Defaults to 1985.  ``int``,
            optional.
        year_end: Last year (inclusive). Defaults to the current year.  ``int``,
            optional.

    Returns:
        Sorted ``list`` of ``str`` holiday names.
    """
    country_holidays = get_available_holidays_in_countries(
        countries=countries,
        year_start=year_start,
        year_end=year_end,
    )
    return sorted({h for h_list in country_holidays.values() for h in h_list})


def build_holiday_features(
    df: "pd.DataFrame",
    time_col: str,
    country_list: List[str],
    years: List[int],
    categories=None,
    per_country: bool = True,
    vicinity_days: int = 0,
) -> "pd.DataFrame":
    """Add per-country holiday binary columns to a timeseries DataFrame.

    Args:
        df: ``pandas.DataFrame`` that must contain a date/datetime column named
            *time_col*.
        time_col: Name of the date column.  ``str``.
        country_list: Country identifiers (ISO-3166 codes or full class names).
            ``list`` of ``str``.
        years: Years to fetch holidays for.  ``list`` of ``int``.
        categories: Passed through to :func:`get_holiday`.  Defaults to PUBLIC
            + OPTIONAL.  ``tuple`` or None, optional.
        per_country: If ``True`` (default), add one binary column per country:
            ``is_holiday_{country.lower()}``.  ``bool``, optional.
        vicinity_days: If > 0, also add ``is_near_holiday_{country.lower()}`` —
            a binary that is 1 on dates within *vicinity_days* days before
            **or** after any holiday in that country, excluding the holiday
            date itself.  ``int``, optional.

    Returns:
        ``pandas.DataFrame`` — copy of *df* with new columns appended.  Original
        columns unchanged.

    Examples:
        >>> df = build_holiday_features(
        ...     df, "ts", ["US", "IN"], years=list(range(2021, 2027)),
        ...     per_country=True, vicinity_days=2,
        ... )
        # always adds:  is_holiday, is_near_holiday  (combined across all countries)
        # per_country:  is_holiday_us, is_holiday_in,
        #               is_near_holiday_us, is_near_holiday_in
    """
    country_holidays = get_holiday(country_list, years, categories=categories)
    df = df.copy()
    dates = pd.to_datetime(df[time_col]).dt.normalize()

    all_on_holiday = pd.Series(False, index=df.index)
    all_near_holiday = pd.Series(False, index=df.index)

    for country, hols in country_holidays.items():
        holiday_dates = pd.DatetimeIndex(pd.to_datetime(list(hols.keys())).normalize())
        col_suffix = country.lower()

        on_holiday = dates.isin(holiday_dates)
        all_on_holiday |= on_holiday

        if per_country:
            df[f"is_holiday_{col_suffix}"] = on_holiday.astype(int)

        if vicinity_days > 0:
            near = pd.Series(False, index=df.index)
            for delta in range(1, vicinity_days + 1):
                near |= dates.isin(holiday_dates - pd.Timedelta(days=delta))
                near |= dates.isin(holiday_dates + pd.Timedelta(days=delta))
            near &= ~on_holiday  # exclude the holiday date itself
            all_near_holiday |= near
            if per_country:
                df[f"is_near_holiday_{col_suffix}"] = near.astype(int)

    # Combined columns — always present regardless of per_country
    df["is_holiday"] = all_on_holiday.astype(int)
    if vicinity_days > 0:
        all_near_holiday &= ~all_on_holiday  # exclude dates that are holidays in any country
        df["is_near_holiday"] = all_near_holiday.astype(int)

    return df
