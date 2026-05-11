import pandas as pd
import pytest
from abvelocity.ts.gk.algo.forecast.silverkite.forecast_simple_silverkite_helper import (
    cols_interact,
    dedup_holiday_dict,
    generate_holiday_events,
    get_event_pred_cols,
    patsy_categorical_term,
    split_events_into_dictionaries,
)
from abvelocity.ts.common.constants import EVENT_DEFAULT, EVENT_DF_DATE_COL, EVENT_DF_LABEL_COL, EVENT_INDICATOR
from abvelocity.ts.common.features.timeseries_features import get_holidays


from abvelocity.ts.gk_test_gate import gk_test_gate

pytestmark = gk_test_gate


def test_cols_interact():
    """Tests cols_interact"""
    columns = cols_interact(static_col="is_weekend", fs_name="hour", fs_order=4)
    assert columns == [
        "is_weekend:sin1_hour",
        "is_weekend:cos1_hour",
        "is_weekend:sin2_hour",
        "is_weekend:cos2_hour",
        "is_weekend:sin3_hour",
        "is_weekend:cos3_hour",
        "is_weekend:sin4_hour",
        "is_weekend:cos4_hour",
    ]

    columns = cols_interact(static_col="is_weekend:ct1", fs_name="hour", fs_order=3, fs_seas_name="seas")
    assert columns == [
        "is_weekend:ct1:sin1_hour_seas",
        "is_weekend:ct1:cos1_hour_seas",
        "is_weekend:ct1:sin2_hour_seas",
        "is_weekend:ct1:cos2_hour_seas",
        "is_weekend:ct1:sin3_hour_seas",
        "is_weekend:ct1:cos3_hour_seas",
    ]


def test_dedup_holiday_dict():
    """Tests dedup_holiday_dict"""
    countries = ["UnitedStates", "UnitedKingdom", "India", "France", "Ireland"]
    year_start = 2019
    year_end = 2020
    # retrieves separate DataFrame for each country, with list of holidays
    holidays_dict = get_holidays(countries, year_start=year_start, year_end=year_end)
    # merges country DataFrames, removes duplicate holidays
    holiday_df = dedup_holiday_dict(holidays_dict)
    assert not holiday_df.duplicated().any()

    # ensure all country holidays are included
    for country, country_df in holidays_dict.items():
        joined = country_df.merge(holiday_df, on=[EVENT_DF_DATE_COL, EVENT_DF_LABEL_COL])
        assert joined.shape[0] == country_df.shape[0]  # checks if all values are contained in holiday_df


def test_split_events_into_dictionaries():
    """Tests split_events_into_dictionaries"""
    countries = ["UnitedStates", "UnitedKingdom", "India", "France"]
    year_start = 2019
    year_end = 2020
    holidays_to_model_separately = ["New Years", "Christmas", "Independence", "Thanksgiving", "Labor", "Good Friday", "Boxing Day", "Memorial", "Veterans"]

    # retrieves separate DataFrame for each country, with list of holidays
    holidays_dict = get_holidays(countries, year_start=year_start, year_end=year_end)
    # merges country DataFrames, removes duplicate holidays
    holiday_df = dedup_holiday_dict(holidays_dict)
    # creates separate DataFrame for each holiday
    daily_event_df_dict = split_events_into_dictionaries(holiday_df, holidays_to_model_separately, default_category="other")

    expected_holidays = [holiday.replace("'", "") for holiday in holidays_to_model_separately] + ["other"]
    assert set(expected_holidays) == set(daily_event_df_dict.keys())

    pd.testing.assert_frame_equal(
        daily_event_df_dict["Christmas"],
        pd.DataFrame({EVENT_DF_DATE_COL: pd.to_datetime(["2019-12-25", "2020-12-25"]), EVENT_DF_LABEL_COL: [EVENT_INDICATOR, EVENT_INDICATOR]}),
        check_dtype=False,
    )

    pd.testing.assert_frame_equal(
        daily_event_df_dict["Boxing Day"],
        pd.DataFrame({EVENT_DF_DATE_COL: pd.to_datetime(["2019-12-26", "2020-12-26"]), EVENT_DF_LABEL_COL: [EVENT_INDICATOR, EVENT_INDICATOR]}),
        check_dtype=False,
    )

    # warns if holiday is not found in the countries
    with pytest.warns(Warning) as record:
        holidays_dict = get_holidays(["UnitedStates", "UnitedKingdom"], year_start=year_start, year_end=year_end)
        holiday_df = dedup_holiday_dict(holidays_dict)
        split_events_into_dictionaries(events_df=holiday_df, events=["New Years", "Unknown Holiday"], default_category="other")
        assert "Requested holiday 'Unknown Holiday' does not occur in the provided countries" in record[0].message.args[0]


def test_generate_holiday_events():
    """Tests generate_holiday_events"""
    countries = ["UnitedStates", "UnitedKingdom", "India", "France"]
    year_start = 2019
    year_end = 2020
    holidays_to_model_separately = ["New Years", "Christmas", "Independence", "Thanksgiving", "Labor", "Good Friday", "Boxing Day", "Memorial", "Veterans"]
    pre_num = 2
    post_num = 2

    daily_event_df_dict = generate_holiday_events(
        countries=countries,
        holidays_to_model_separately=holidays_to_model_separately,
        year_start=year_start,
        year_end=year_end,
        pre_num=pre_num,
        post_num=post_num,
    )

    cleaned_holidays = [holiday.replace("'", "") for holiday in holidays_to_model_separately]
    cleaned_holidays += ["Other"]  # default value is "Other"
    expected_holidays = cleaned_holidays.copy()
    for i in range(1, pre_num + 1):
        expected_holidays += [f"{holiday}_minus_{i}" for holiday in cleaned_holidays]
    for i in range(1, post_num + 1):
        expected_holidays += [f"{holiday}_plus_{i}" for holiday in cleaned_holidays]

    assert set(expected_holidays) == set(list(daily_event_df_dict.keys()))

    pd.testing.assert_frame_equal(
        daily_event_df_dict["Christmas"],
        pd.DataFrame({EVENT_DF_DATE_COL: pd.to_datetime(["2019-12-25", "2020-12-25"]), EVENT_DF_LABEL_COL: [EVENT_INDICATOR, EVENT_INDICATOR]}),
        check_dtype=False,
    )

    pd.testing.assert_frame_equal(
        daily_event_df_dict["Boxing Day_plus_1"],
        pd.DataFrame({EVENT_DF_DATE_COL: pd.to_datetime(["2019-12-27", "2020-12-27"]), EVENT_DF_LABEL_COL: [EVENT_INDICATOR, EVENT_INDICATOR]}),
        check_dtype=False,
    )

    pd.testing.assert_frame_equal(
        daily_event_df_dict["Veterans_minus_2"],
        pd.DataFrame({EVENT_DF_DATE_COL: pd.to_datetime(["2019-11-09", "2020-11-09"]), EVENT_DF_LABEL_COL: [EVENT_INDICATOR, EVENT_INDICATOR]}),
        check_dtype=False,
    )


def test_generate_holiday_events2():
    """Tests proper handling of pre_num = 0 and post_num = 0"""
    countries = ["UnitedStates", "UnitedKingdom", "India", "France"]
    year_start = 2019
    year_end = 2020
    holidays_to_model_separately = ["New Years", "Christmas", "Independence", "Thanksgiving", "Labor", "Good Friday", "Boxing Day", "Memorial", "Veterans"]

    daily_event_df_dict1 = generate_holiday_events(
        countries=countries, holidays_to_model_separately=holidays_to_model_separately, year_start=year_start, year_end=year_end, pre_num=0, post_num=0
    )

    holidays_dict = get_holidays(countries, year_start=year_start, year_end=year_end)
    # merges country DataFrames, removes duplicate holidays
    holiday_df = dedup_holiday_dict(holidays_dict)
    # creates separate DataFrame for each holiday
    daily_event_df_dict2 = split_events_into_dictionaries(holiday_df, holidays_to_model_separately)

    assert daily_event_df_dict1.keys() == daily_event_df_dict2.keys()


def test_generate_holiday_events3():
    """Tests generate_holiday_events pre_post_num_dict parameter"""
    # Tests pre_post_num_dict
    countries = ["UnitedStates", "India"]
    year_start = 2019
    year_end = 2020
    holidays_to_model_separately = ["New Years", "Diwali", "Columbus"]
    pre_num = 2
    post_num = 2
    pre_post_num_dict = {"New Years": (0, 2), "Columbus": (1, 3)}
    daily_event_df_dict = generate_holiday_events(
        countries=countries,
        holidays_to_model_separately=holidays_to_model_separately,
        year_start=year_start,
        year_end=year_end,
        pre_num=pre_num,
        post_num=post_num,
        pre_post_num_dict=pre_post_num_dict,
    )

    # expected
    expected_holidays = ["New Years_plus_2", "Diwali_minus_2", "Diwali_plus_2", "Columbus_minus_1", "Columbus_plus_3"]
    assert all([holiday in daily_event_df_dict.keys() for holiday in expected_holidays])
    unexpected_holidays = ["New Years_minus_1", "New Years_plus_3", "Diwali_minus_3", "Diwali_plus_3", "Columbus_minus_2", "Columbus_plus_4"]
    assert not any([holiday in daily_event_df_dict.keys() for holiday in unexpected_holidays])

    with pytest.warns(UserWarning) as record:
        pre_post_num_dict = {"Bank Holiday": (1, 1)}
        generate_holiday_events(
            countries=countries,
            holidays_to_model_separately=holidays_to_model_separately,
            year_start=year_start,
            year_end=year_end,
            pre_num=pre_num,
            post_num=post_num,
            pre_post_num_dict=pre_post_num_dict,
        )
        all_warning_msg = " ".join([record[i].message.args[0] for i in range(len(record))])
        assert "Requested holiday 'Bank Holiday' is not valid. Valid holidays are" in all_warning_msg


def test_patsy_categorical_term():
    """Tests patsy_categorical_term"""
    term = patsy_categorical_term("term", levels=None, coding=None, quote=False)
    assert term == "C(term)"

    term = patsy_categorical_term("term", levels=None, coding=None, quote=True)
    assert term == "C(Q('term'))"

    term = patsy_categorical_term("term", levels=None, coding="Sum", quote=True)
    assert term == "C(Q('term'), Sum)"

    term = patsy_categorical_term("term", levels=["level1", "level2"], coding="Sum", quote=True)
    assert term == "C(Q('term'), Sum, levels=['level1', 'level2'])"

    term = patsy_categorical_term("term", levels=["level1", "level2"], coding="Sum", quote=False)
    assert term == "C(term, Sum, levels=['level1', 'level2'])"


def test_get_event_pred_cols():
    """Tests get_event_pred_cols"""
    assert get_event_pred_cols(None) == []

    daily_event_df_dict = {
        "k1": pd.DataFrame({EVENT_DF_LABEL_COL: ["level1", "level2", "level1"]}),
        "k2": pd.DataFrame({EVENT_DF_LABEL_COL: ["a", "b", "c"]}),
        "k3": pd.DataFrame({EVENT_DF_LABEL_COL: ["d", "d", "d"]}),
    }
    assert get_event_pred_cols(daily_event_df_dict) == [
        f"C(Q('events_k1'), levels=['{EVENT_DEFAULT}', 'level1', 'level2'])",
        f"C(Q('events_k2'), levels=['{EVENT_DEFAULT}', 'a', 'b', 'c'])",
        f"C(Q('events_k3'), levels=['{EVENT_DEFAULT}', 'd'])",
    ]


def test_get_event_pred_cols_with_neighbor():
    """Tests `get_event_pred_cols` with neighbor events."""

    daily_event_df_dict = {
        "k1": pd.DataFrame({EVENT_DF_LABEL_COL: ["level1", "level2", "level1"]}),
        "k2": pd.DataFrame({EVENT_DF_LABEL_COL: ["a", "b", "c"]}),
        "k3": pd.DataFrame({EVENT_DF_LABEL_COL: ["d", "d", "d"]}),
    }
    assert get_event_pred_cols(daily_event_df_dict, ["7D"]) == [
        f"C(Q('events_k1'), levels=['{EVENT_DEFAULT}', 'level1', 'level2'])",
        f"C(Q('events_k1_7D_after'), levels=['{EVENT_DEFAULT}', 'level1_7D_after', 'level2_7D_after'])",
        f"C(Q('events_k2'), levels=['{EVENT_DEFAULT}', 'a', 'b', 'c'])",
        f"C(Q('events_k2_7D_after'), levels=['{EVENT_DEFAULT}', 'a_7D_after', 'b_7D_after', 'c_7D_after'])",
        f"C(Q('events_k3'), levels=['{EVENT_DEFAULT}', 'd'])",
        f"C(Q('events_k3_7D_after'), levels=['{EVENT_DEFAULT}', 'd_7D_after'])",
    ]
