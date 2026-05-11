"""Tests for abvelocity.ts.common.holiday.get_holidays.

Each test fetches the full holiday list for a country + year and compares
it exactly against the hardcoded list below.  Reading the lists is the
fastest way to verify coverage before a forecast run.

Name conventions visible in the lists (applied by get_holiday automatically):
  - Trailing " Day" stripped: "Christmas" → "Christmas", "Labor" → "Labor",
    "Independence" → "Independence", "Republic Day" → "Republic", etc.
    Exceptions kept as-is: "Boxing Day", "May Day", "National Day".
  - Periods removed: "MLK" → "MLK" (via canonical).
  - Accents stripped: "Atatürk" → "Ataturk".
  - Parentheticals removed: "(observed)" / "(estimated)" stripped.
  - 4-word limit, then canonical abbreviations:
      "Martin Luther King Jr" → "MLK"
      "Universal Fraternization" → "New Years"  (Brazil Jan 1)
      "Workers" / "Labour and Solidarity" / "Labour" → "Labor"
      "National Day of Zumbi" → "Black Awareness"  (Brazil Nov 20)
      "Commemoration of Ataturk Youth" → "Ataturk"  (Turkey May 19)

Year-over-year consistency notes:
  US  — 12 entries every year (11 federal + Halloween); only dates shift.
        2026: Jul 4 is Sat, so observed entry on Jul 3 — both labelled "Independence".
  IN  — ~27 entries; count varies because Islamic holidays are lunar-computed.
        When two festivals share a date (e.g. 2025-10-02), only the first is stored.
  GB  — exactly 6 entries per year; Easter-anchored dates shift.
        2026: Boxing Day Sat → extra observed entry on Mon Dec 28.
  BR  — exactly 17 entries; Carnival always spans two consecutive days.
  CA  — 6 entries (federal only); 2026 has extra Boxing Day observed entry.
  FR  — 11 entries; Easter Mon / Ascension / Whit Mon dates shift each year.
"""

import pandas as pd
from abvelocity.ts.common.holiday.get_holidays import build_holiday_features, get_holiday

from abvelocity.ts.gk_test_gate import gk_test_gate

pytestmark = gk_test_gate


def snapshot_check(country_code, expected_list, result):
    """Assert exact date→name pairs match the expected list."""
    expected = {(d, n) for d, n in expected_list}
    actual = {(str(d), n) for d, n in result[country_code].items()}
    missing = expected - actual
    extra = actual - expected
    assert not missing, f"[{country_code}] Missing: {sorted(missing)}"
    assert not extra, f"[{country_code}] Unexpected: {sorted(extra)}"


# ─────────────────────────────────────────────────────────────────────────────
# United States  (PUBLIC only — no OPTIONAL support; Halloween added by greykite)
# ─────────────────────────────────────────────────────────────────────────────

US_2024 = [
    ("2024-01-01", "New Years"),
    ("2024-01-15", "MLK"),
    ("2024-02-19", "Washingtons Birthday"),
    ("2024-05-27", "Memorial"),
    ("2024-06-19", "Juneteenth National Independence"),
    ("2024-07-04", "Independence"),
    ("2024-09-02", "Labor"),
    ("2024-10-14", "Columbus"),
    ("2024-10-31", "Halloween"),
    ("2024-11-11", "Veterans"),
    ("2024-11-28", "Thanksgiving"),
    ("2024-12-25", "Christmas"),
]

US_2025 = [
    ("2025-01-01", "New Years"),
    ("2025-01-20", "MLK"),
    ("2025-02-17", "Washingtons Birthday"),
    ("2025-05-26", "Memorial"),
    ("2025-06-19", "Juneteenth National Independence"),
    ("2025-07-04", "Independence"),
    ("2025-09-01", "Labor"),
    ("2025-10-13", "Columbus"),
    ("2025-10-31", "Halloween"),
    ("2025-11-11", "Veterans"),
    ("2025-11-27", "Thanksgiving"),
    ("2025-12-25", "Christmas"),
]

US_2026 = [
    # Jul 4 is Sat → observed Fri Jul 3 carries "Independence_observed"
    ("2026-01-01", "New Years"),
    ("2026-01-19", "MLK"),
    ("2026-02-16", "Washingtons Birthday"),
    ("2026-05-25", "Memorial"),
    ("2026-06-19", "Juneteenth National Independence"),
    ("2026-07-03", "Independence_observed"),
    ("2026-07-04", "Independence"),
    ("2026-09-07", "Labor"),
    ("2026-10-12", "Columbus"),
    ("2026-10-31", "Halloween"),
    ("2026-11-11", "Veterans"),
    ("2026-11-26", "Thanksgiving"),
    ("2026-12-25", "Christmas"),
]


def test_snapshot_us_2024():
    snapshot_check("US", US_2024, get_holiday(["US"], years=[2024]))


def test_snapshot_us_2025():
    snapshot_check("US", US_2025, get_holiday(["US"], years=[2025]))


def test_snapshot_us_2026():
    snapshot_check("US", US_2026, get_holiday(["US"], years=[2026]))


# ─────────────────────────────────────────────────────────────────────────────
# India  (PUBLIC + OPTIONAL — Holi included)
# ─────────────────────────────────────────────────────────────────────────────

IN_2024 = [
    ("2024-01-14", "Makar Sankranti"),
    ("2024-01-26", "Republic"),
    ("2024-03-08", "Maha Shivaratri"),
    ("2024-03-24", "Palm Sunday"),
    ("2024-03-25", "Holi"),
    ("2024-03-29", "Good Friday"),
    ("2024-03-31", "Easter Sunday"),
    ("2024-04-11", "Eid al-Fitr"),
    ("2024-04-17", "Ram Navami"),
    ("2024-04-21", "Mahavir Jayanti"),
    ("2024-05-01", "Labor"),
    ("2024-05-23", "Buddha Purnima"),
    ("2024-06-17", "Eid al-Adha"),
    ("2024-07-17", "Ashura"),
    ("2024-08-15", "Independence"),
    ("2024-08-19", "Raksha Bandhan"),
    ("2024-08-26", "Janmashtami"),
    ("2024-09-07", "Ganesh Chaturthi"),
    ("2024-09-16", "Prophets Birthday"),
    ("2024-10-02", "Gandhi Jayanti"),
    ("2024-10-03", "Navratri / Sharad Navratri"),
    ("2024-10-11", "Maha Navami"),
    ("2024-10-12", "Dussehra"),
    ("2024-10-31", "Diwali"),
    ("2024-11-02", "Govardhan Puja"),
    ("2024-11-14", "Childrens"),
    ("2024-11-15", "Guru Nanak Jayanti"),
    ("2024-12-25", "Christmas"),
]

IN_2025 = [
    ("2025-01-14", "Makar Sankranti"),
    ("2025-01-26", "Republic"),
    ("2025-02-26", "Maha Shivaratri"),
    ("2025-03-14", "Holi"),
    ("2025-03-31", "Eid al-Fitr"),
    ("2025-04-06", "Ram Navami"),
    ("2025-04-10", "Mahavir Jayanti"),
    ("2025-04-13", "Palm Sunday"),
    ("2025-04-18", "Good Friday"),
    ("2025-04-20", "Easter Sunday"),
    ("2025-05-01", "Labor"),
    ("2025-05-12", "Buddha Purnima"),
    ("2025-06-07", "Eid al-Adha"),
    ("2025-07-06", "Ashura"),
    ("2025-08-09", "Raksha Bandhan"),
    ("2025-08-15", "Independence"),
    ("2025-08-16", "Janmashtami"),
    ("2025-08-27", "Ganesh Chaturthi"),
    ("2025-09-05", "Prophets Birthday"),
    ("2025-09-22", "Navratri / Sharad Navratri"),
    ("2025-10-01", "Maha Navami"),
    # 2025-10-02: Dussehra and Gandhi Jayanti share this date; only Dussehra stored
    ("2025-10-02", "Dussehra"),
    ("2025-10-20", "Diwali"),
    ("2025-10-22", "Govardhan Puja"),
    ("2025-11-05", "Guru Nanak Jayanti"),
    ("2025-11-14", "Childrens"),
    ("2025-12-25", "Christmas"),
]

IN_2026 = [
    ("2026-01-14", "Makar Sankranti"),
    ("2026-01-26", "Republic"),
    ("2026-02-15", "Maha Shivaratri"),
    ("2026-03-04", "Holi"),
    ("2026-03-21", "Eid al-Fitr"),  # (estimated) stripped
    ("2026-03-26", "Ram Navami"),
    ("2026-03-29", "Palm Sunday"),
    ("2026-03-31", "Mahavir Jayanti"),
    ("2026-04-03", "Good Friday"),
    ("2026-04-05", "Easter Sunday"),
    # 2026-05-01: Buddha Purnima and Labor share this date; only Buddha Purnima stored
    ("2026-05-01", "Buddha Purnima"),
    ("2026-05-28", "Eid al-Adha"),  # (estimated) stripped
    ("2026-06-26", "Ashura"),  # (estimated) stripped
    ("2026-08-15", "Independence"),
    ("2026-08-26", "Prophets Birthday"),
    ("2026-08-28", "Raksha Bandhan"),
    ("2026-09-04", "Janmashtami"),
    ("2026-09-14", "Ganesh Chaturthi"),
    ("2026-10-02", "Gandhi Jayanti"),
    ("2026-10-11", "Navratri / Sharad Navratri"),
    ("2026-10-19", "Maha Navami"),
    ("2026-10-20", "Dussehra"),
    ("2026-11-08", "Diwali"),
    ("2026-11-10", "Govardhan Puja"),
    ("2026-11-14", "Childrens"),
    ("2026-11-24", "Guru Nanak Jayanti"),
    ("2026-12-25", "Christmas"),
]


def test_snapshot_in_2024():
    snapshot_check("IN", IN_2024, get_holiday(["IN"], years=[2024]))


def test_snapshot_in_2025():
    snapshot_check("IN", IN_2025, get_holiday(["IN"], years=[2025]))


def test_snapshot_in_2026():
    snapshot_check("IN", IN_2026, get_holiday(["IN"], years=[2026]))


# ─────────────────────────────────────────────────────────────────────────────
# Great Britain  (PUBLIC only — no OPTIONAL defined for GB)
# "May Day" and "Boxing Day" keep their " Day" suffix (exception list).
# ─────────────────────────────────────────────────────────────────────────────

GB_2024 = [
    ("2024-01-01", "New Years"),
    ("2024-03-29", "Good Friday"),
    ("2024-05-06", "May Day"),
    ("2024-05-27", "Spring Bank Holiday"),
    ("2024-12-25", "Christmas"),
    ("2024-12-26", "Boxing Day"),
]

GB_2025 = [
    ("2025-01-01", "New Years"),
    ("2025-04-18", "Good Friday"),
    ("2025-05-05", "May Day"),
    ("2025-05-26", "Spring Bank Holiday"),
    ("2025-12-25", "Christmas"),
    ("2025-12-26", "Boxing Day"),
]

GB_2026 = [
    # Boxing Day Sat Dec 26 → observed Mon Dec 28 carries "Boxing Day_observed"
    ("2026-01-01", "New Years"),
    ("2026-04-03", "Good Friday"),
    ("2026-05-04", "May Day"),
    ("2026-05-25", "Spring Bank Holiday"),
    ("2026-12-25", "Christmas"),
    ("2026-12-26", "Boxing Day"),
    ("2026-12-28", "Boxing Day_observed"),
]


def test_snapshot_gb_2024():
    snapshot_check("GB", GB_2024, get_holiday(["GB"], years=[2024]))


def test_snapshot_gb_2025():
    snapshot_check("GB", GB_2025, get_holiday(["GB"], years=[2025]))


def test_snapshot_gb_2026():
    snapshot_check("GB", GB_2026, get_holiday(["GB"], years=[2026]))


# ─────────────────────────────────────────────────────────────────────────────
# Brazil  (PUBLIC + OPTIONAL — Carnival included)
# "Universal Fraternization" → "New Years" (canonical)
# "Workers" → "Labor" (canonical)
# "National Day of Zumbi" → "Black Awareness" (canonical)
# ─────────────────────────────────────────────────────────────────────────────

BR_2024 = [
    ("2024-01-01", "New Years"),
    ("2024-02-12", "Carnival"),
    ("2024-02-13", "Carnival"),
    ("2024-02-14", "Ash Wednesday"),
    ("2024-03-29", "Good Friday"),
    ("2024-04-21", "Tiradentes"),
    ("2024-05-01", "Labor"),
    ("2024-05-30", "Corpus Christi"),
    ("2024-09-07", "Independence"),
    ("2024-10-12", "Our Lady of Aparecida"),
    ("2024-10-28", "Public Servants"),
    ("2024-11-02", "All Souls"),
    ("2024-11-15", "Republic Proclamation"),
    ("2024-11-20", "Black Awareness"),
    ("2024-12-24", "Christmas Eve"),
    ("2024-12-25", "Christmas"),
    ("2024-12-31", "New Years Eve"),
]

BR_2025 = [
    ("2025-01-01", "New Years"),
    ("2025-03-03", "Carnival"),
    ("2025-03-04", "Carnival"),
    ("2025-03-05", "Ash Wednesday"),
    ("2025-04-18", "Good Friday"),
    ("2025-04-21", "Tiradentes"),
    ("2025-05-01", "Labor"),
    ("2025-06-19", "Corpus Christi"),
    ("2025-09-07", "Independence"),
    ("2025-10-12", "Our Lady of Aparecida"),
    ("2025-10-28", "Public Servants"),
    ("2025-11-02", "All Souls"),
    ("2025-11-15", "Republic Proclamation"),
    ("2025-11-20", "Black Awareness"),
    ("2025-12-24", "Christmas Eve"),
    ("2025-12-25", "Christmas"),
    ("2025-12-31", "New Years Eve"),
]

BR_2026 = [
    ("2026-01-01", "New Years"),
    ("2026-02-16", "Carnival"),
    ("2026-02-17", "Carnival"),
    ("2026-02-18", "Ash Wednesday"),
    ("2026-04-03", "Good Friday"),
    ("2026-04-21", "Tiradentes"),
    ("2026-05-01", "Labor"),
    ("2026-06-04", "Corpus Christi"),
    ("2026-09-07", "Independence"),
    ("2026-10-12", "Our Lady of Aparecida"),
    ("2026-10-28", "Public Servants"),
    ("2026-11-02", "All Souls"),
    ("2026-11-15", "Republic Proclamation"),
    ("2026-11-20", "Black Awareness"),
    ("2026-12-24", "Christmas Eve"),
    ("2026-12-25", "Christmas"),
    ("2026-12-31", "New Years Eve"),
]


def test_snapshot_br_2024():
    snapshot_check("BR", BR_2024, get_holiday(["BR"], years=[2024]))


def test_snapshot_br_2025():
    snapshot_check("BR", BR_2025, get_holiday(["BR"], years=[2025]))


def test_snapshot_br_2026():
    snapshot_check("BR", BR_2026, get_holiday(["BR"], years=[2026]))


# ─────────────────────────────────────────────────────────────────────────────
# Canada  (federal PUBLIC holidays only — 6 per year)
# "Boxing Day" kept (exception); "Canada Day" → "Canada".
# ─────────────────────────────────────────────────────────────────────────────

CA_2024 = [
    ("2024-01-01", "New Years"),
    ("2024-03-29", "Good Friday"),
    ("2024-07-01", "Canada"),
    ("2024-09-02", "Labor"),
    ("2024-12-25", "Christmas"),
    ("2024-12-26", "Boxing Day"),
]

CA_2025 = [
    ("2025-01-01", "New Years"),
    ("2025-04-18", "Good Friday"),
    ("2025-07-01", "Canada"),
    ("2025-09-01", "Labor"),
    ("2025-12-25", "Christmas"),
    ("2025-12-26", "Boxing Day"),
]

CA_2026 = [
    # Boxing Day Sat Dec 26 → observed Mon Dec 28 carries "Boxing Day_observed"
    ("2026-01-01", "New Years"),
    ("2026-04-03", "Good Friday"),
    ("2026-07-01", "Canada"),
    ("2026-09-07", "Labor"),
    ("2026-12-25", "Christmas"),
    ("2026-12-26", "Boxing Day"),
    ("2026-12-28", "Boxing Day_observed"),
]


def test_snapshot_ca_2024():
    snapshot_check("CA", CA_2024, get_holiday(["CA"], years=[2024]))


def test_snapshot_ca_2025():
    snapshot_check("CA", CA_2025, get_holiday(["CA"], years=[2025]))


def test_snapshot_ca_2026():
    snapshot_check("CA", CA_2026, get_holiday(["CA"], years=[2026]))


# ─────────────────────────────────────────────────────────────────────────────
# France  (11 entries; Easter-anchored dates shift each year)
# "National Day" kept (exception); "Ascension Day" → "Ascension" etc.
# ─────────────────────────────────────────────────────────────────────────────

FR_2024 = [
    ("2024-01-01", "New Years"),
    ("2024-04-01", "Easter Monday"),
    ("2024-05-01", "Labor"),
    ("2024-05-08", "Victory"),
    ("2024-05-09", "Ascension"),
    ("2024-05-20", "Whit Monday"),
    ("2024-07-14", "National Day"),
    ("2024-08-15", "Assumption"),
    ("2024-11-01", "All Saints"),
    ("2024-11-11", "Armistice"),
    ("2024-12-25", "Christmas"),
]

FR_2025 = [
    ("2025-01-01", "New Years"),
    ("2025-04-21", "Easter Monday"),
    ("2025-05-01", "Labor"),
    ("2025-05-08", "Victory"),
    ("2025-05-29", "Ascension"),
    ("2025-06-09", "Whit Monday"),
    ("2025-07-14", "National Day"),
    ("2025-08-15", "Assumption"),
    ("2025-11-01", "All Saints"),
    ("2025-11-11", "Armistice"),
    ("2025-12-25", "Christmas"),
]

FR_2026 = [
    ("2026-01-01", "New Years"),
    ("2026-04-06", "Easter Monday"),
    ("2026-05-01", "Labor"),
    ("2026-05-08", "Victory"),
    ("2026-05-14", "Ascension"),
    ("2026-05-25", "Whit Monday"),
    ("2026-07-14", "National Day"),
    ("2026-08-15", "Assumption"),
    ("2026-11-01", "All Saints"),
    ("2026-11-11", "Armistice"),
    ("2026-12-25", "Christmas"),
]


def test_snapshot_fr_2024():
    snapshot_check("FR", FR_2024, get_holiday(["FR"], years=[2024]))


def test_snapshot_fr_2025():
    snapshot_check("FR", FR_2025, get_holiday(["FR"], years=[2025]))


def test_snapshot_fr_2026():
    snapshot_check("FR", FR_2026, get_holiday(["FR"], years=[2026]))


# ─────────────────────────────────────────────────────────────────────────────
# build_holiday_features
# ─────────────────────────────────────────────────────────────────────────────


def make_date_df(start, end):
    """Helper: daily DataFrame with a 'ts' column."""
    return pd.DataFrame({"ts": pd.date_range(start, end, freq="D")})


def test_build_holiday_features_combined_columns():
    """is_holiday and is_near_holiday (combined) are always written."""
    df = make_date_df("2024-12-24", "2025-01-03")
    result = build_holiday_features(df, "ts", ["US"], years=[2024, 2025], per_country=False, vicinity_days=2)
    assert "is_holiday" in result.columns
    assert "is_near_holiday" in result.columns
    # Dec 25 is Christmas — should be 1
    assert result.loc[result["ts"] == "2024-12-25", "is_holiday"].values[0] == 1
    # Dec 26 is within 1 day of Christmas — near but not a holiday itself
    assert result.loc[result["ts"] == "2024-12-26", "is_near_holiday"].values[0] == 1
    assert result.loc[result["ts"] == "2024-12-26", "is_holiday"].values[0] == 0
    # Jan 1 is New Years — holiday, so NOT near
    assert result.loc[result["ts"] == "2025-01-01", "is_holiday"].values[0] == 1
    assert result.loc[result["ts"] == "2025-01-01", "is_near_holiday"].values[0] == 0


def test_build_holiday_features_per_country_columns():
    """per_country=True creates is_holiday_{cc} and is_near_holiday_{cc} columns."""
    df = make_date_df("2025-07-01", "2025-07-16")
    result = build_holiday_features(df, "ts", ["US", "FR"], years=[2025], per_country=True, vicinity_days=1)
    # Per-country binary columns present
    assert "is_holiday_us" in result.columns
    assert "is_holiday_fr" in result.columns
    assert "is_near_holiday_us" in result.columns
    assert "is_near_holiday_fr" in result.columns
    # Jul 4 = US Independence; not a French holiday
    assert result.loc[result["ts"] == "2025-07-04", "is_holiday_us"].values[0] == 1
    assert result.loc[result["ts"] == "2025-07-04", "is_holiday_fr"].values[0] == 0
    # Jul 14 = French National Day; not a US holiday
    assert result.loc[result["ts"] == "2025-07-14", "is_holiday_fr"].values[0] == 1
    assert result.loc[result["ts"] == "2025-07-14", "is_holiday_us"].values[0] == 0


def test_build_holiday_features_no_per_country():
    """per_country=False suppresses per-country columns."""
    df = make_date_df("2025-07-01", "2025-07-10")
    result = build_holiday_features(df, "ts", ["US", "FR"], years=[2025], per_country=False, vicinity_days=0)
    assert "is_holiday" in result.columns
    assert "is_near_holiday" not in result.columns
    assert "is_holiday_us" not in result.columns
    assert "is_holiday_fr" not in result.columns


def test_build_holiday_features_vicinity_zero():
    """vicinity_days=0 means no is_near_holiday columns at all."""
    df = make_date_df("2025-12-24", "2025-12-27")
    result = build_holiday_features(df, "ts", ["US"], years=[2025], per_country=True, vicinity_days=0)
    assert "is_holiday" in result.columns
    assert "is_near_holiday" not in result.columns
    assert "is_near_holiday_us" not in result.columns


def test_build_holiday_features_combined_or_logic():
    """is_holiday is 1 if any country has a holiday on that date."""
    df = make_date_df("2025-07-04", "2025-07-04")
    result = build_holiday_features(df, "ts", ["US", "FR"], years=[2025], per_country=True, vicinity_days=0)
    # Jul 4 is a US holiday — combined column should be 1 even though FR=0
    assert result.loc[result["ts"] == "2025-07-04", "is_holiday"].values[0] == 1
    assert result.loc[result["ts"] == "2025-07-04", "is_holiday_us"].values[0] == 1
    assert result.loc[result["ts"] == "2025-07-04", "is_holiday_fr"].values[0] == 0


def test_build_holiday_features_preserves_original_columns():
    """Original DataFrame columns must be unchanged after the call."""
    df = make_date_df("2025-01-01", "2025-01-05")
    df["value"] = range(5)
    result = build_holiday_features(df, "ts", ["US"], years=[2025], per_country=True, vicinity_days=1)
    assert list(result["value"]) == list(df["value"])
    # Original df should be unmodified (function works on a copy)
    assert "is_holiday" not in df.columns
