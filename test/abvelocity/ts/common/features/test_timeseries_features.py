import datetime
from datetime import datetime as dt
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
from abvelocity.ts.common.constants import (
    CHANGEPOINT_COL_PREFIX,
    EVENT_DF_DATE_COL,
    EVENT_DF_LABEL_COL,
    EVENT_PREFIX,
    IS_EVENT_ADJACENT_COL,
    IS_EVENT_COL,
    IS_EVENT_EXACT_COL,
    TIME_COL,
)
from abvelocity.ts.common.features.timeseries_features import (
    add_daily_events,
    add_event_window,
    add_event_window_multi,
    add_time_features_df,
    build_time_features_df,
    convert_date_to_continuous_time,
    fourier_series_fcn,
    fourier_series_multi_fcn,
    get_available_holiday_lookup_countries,
    get_available_holidays_across_countries,
    get_available_holidays_in_countries,
    get_changepoint_dates_from_changepoints_dict,
    get_changepoint_features,
    get_changepoint_features_and_values_from_config,
    get_changepoint_string,
    get_changepoint_values_from_config,
    get_custom_changepoints_values,
    get_default_origin_for_time_vars,
    get_eu_dst_end,
    get_eu_dst_start,
    get_evenly_spaced_changepoints_dates,
    get_evenly_spaced_changepoints_values,
    get_fourier_col_name,
    get_holidays,
    get_logistic_func,
    get_us_dst_end,
    get_us_dst_start,
    is_dst_fcn,
    logistic,
    pytz_is_dst_fcn,
    signed_pow,
    signed_pow_fcn,
    signed_sqrt,
)
from abvelocity.ts.common.python_utils import assert_equal
from abvelocity.ts.common.testing_utils import generate_df_for_tests, generate_df_with_reg_for_tests

from abvelocity.ts.gk_test_gate import gk_test_gate

pytestmark = gk_test_gate


@pytest.fixture
def hourly_data():
    """Generate 500 days of hourly data for tests"""
    return generate_df_for_tests(freq="H", periods=24 * 500)


# Below defines a function to do all tests for daylight saving functions
def validate_dst_times(is_dst_func, dst_start, dst_end):
    """Assuming dayligh saving starts on ``dst_start`` and ends on ``dst_end``,
    it performs a series of assertion to ensure the function
    ``is_dst_func`` works as expected."""
    # It must be daylight saving on the start date
    dt = dst_start
    assert is_dst_func([dt])[0] is True, "dst start not correct"
    # It must be still daylight saving an hour after the start date
    assert is_dst_func([dt + pd.to_timedelta("1H")])[0] is True, "an hour after start of dst"
    # It must be still daylight saving 2 hours after the start date
    assert is_dst_func([dt + pd.to_timedelta("2H")])[0] is True, "2 hours after start of dst"
    # It must be still daylight saving 90 days after the start date
    assert is_dst_func([dt + pd.to_timedelta("90D")])[0] is True, "90 days after start of dst"
    # It is not daylight saving an hour before the start date
    assert is_dst_func([dt - pd.to_timedelta("1H")])[0] is False, "an hour before start of dst"

    # It must be daylight saving right before the end date
    dt = dst_end
    assert is_dst_func([dt])[0] is True
    # It must be still daylight saving an hour before the end date
    assert is_dst_func([dt - pd.to_timedelta("1H")])[0] is True, "an hour before end of dst"
    # It must be still daylight saving 2 hours before the end date
    assert is_dst_func([dt - pd.to_timedelta("2H")])[0] is True, "2 hours before end of dst"
    # It must be still daylight saving 90 days before the end date
    assert is_dst_func([dt - pd.to_timedelta("90D")])[0] is True, "90 days before end of dst"
    # It is not daylight saving an hour after the end date
    assert is_dst_func([dt + pd.to_timedelta("1H")])[0] is False, "an hour the end of dst"


def test_convert_date_to_continuous_time():
    assert convert_date_to_continuous_time(dt(2019, 1, 1)) == 2019.0
    assert convert_date_to_continuous_time(dt(2019, 7, 1)) == 2019 + 181 / 365
    assert convert_date_to_continuous_time(dt(2020, 7, 1)) == 2020 + 182 / 366  # leap year
    assert convert_date_to_continuous_time(dt(2019, 7, 1, 7, 4, 24)) == (
        2019 + (181 + 7 / 24 + 4 / (24 * 60) + 24 / (24 * 60 * 60)) / 365  # day  # hour  # minute  # second
    )


def test_get_default_origin_for_time_vars(hourly_data):
    """Tests get_default_origin_for_time_vars"""
    train_df = hourly_data["train_df"]
    conti_year_origin = get_default_origin_for_time_vars(train_df, TIME_COL)
    assert round(conti_year_origin, 3) == 2018.496

    df = pd.DataFrame({"time": ["2018-07-01", "2018-08-01"]})
    conti_year_origin = get_default_origin_for_time_vars(df, "time")
    assert round(conti_year_origin, 3) == 2018.496


def test_pytz_is_dst_fcn():
    """Tests ``pytz_is_dst_fcn``"""
    # Tests the function constructed for US
    # Note that as long as the zone is inside mainland US,
    # result should be the same
    us_dst_start = pd.to_datetime("2022-03-13 02:00:00")
    us_dst_end = pd.to_datetime("2022-11-06 01:59:59")
    validate_dst_times(is_dst_func=pytz_is_dst_fcn("US/Pacific"), dst_start=us_dst_start, dst_end=us_dst_end)

    # Repeates the US case, with another time zone
    validate_dst_times(is_dst_func=pytz_is_dst_fcn("US/Central"), dst_start=us_dst_start, dst_end=us_dst_end)

    # Europe case
    eu_dst_start = pd.to_datetime("2022-03-27 01:00:00")
    eu_dst_end = pd.to_datetime("2022-10-30 01:59:59")
    validate_dst_times(is_dst_func=pytz_is_dst_fcn("Europe/London"), dst_start=eu_dst_start, dst_end=eu_dst_end)


def test_is_dst_fcn():
    """Tests ``is_dst_fcn``"""
    # Tests the function constructed for US
    # Note that as long as the zone is inside mainland US,
    # result should be the same
    us_dst_start = pd.to_datetime("2022-03-13 02:00:00")
    us_dst_end = pd.to_datetime("2022-11-06 01:59:59")
    validate_dst_times(is_dst_func=is_dst_fcn("US/Pacific"), dst_start=us_dst_start, dst_end=us_dst_end)

    # Repeates the US case, with another time zone
    validate_dst_times(is_dst_func=is_dst_fcn("US/Central"), dst_start=us_dst_start, dst_end=us_dst_end)

    # Europe case
    eu_dst_start = pd.to_datetime("2022-03-27 01:00:00")
    eu_dst_end = pd.to_datetime("2022-10-30 01:59:59")
    validate_dst_times(is_dst_func=is_dst_fcn("Europe/London"), dst_start=eu_dst_start, dst_end=eu_dst_end)


def test_get_dst_start_end_date():
    """Tests `get_us_dst_start`, `get_us_dst_end`,
    `get_eu_dst_start`, `get_eu_dst_end` functions.
    """
    years = [2015, 2022, 2023, 2024]
    expected_us_dst_start_dates = [dt(2015, 3, 8, 2, 0), dt(2022, 3, 13, 2, 0), dt(2023, 3, 12, 2, 0), dt(2024, 3, 10, 2, 0)]
    expected_us_dst_end_dates = [dt(2015, 11, 1, 2, 0), dt(2022, 11, 6, 2, 0), dt(2023, 11, 5, 2, 0), dt(2024, 11, 3, 2, 0)]
    expected_eu_dst_start_dates = [dt(2015, 3, 29, 1, 0), dt(2022, 3, 27, 1, 0), dt(2023, 3, 26, 1, 0), dt(2024, 3, 31, 1, 0)]
    expected_eu_dst_end_dates = [dt(2015, 10, 25, 2, 0), dt(2022, 10, 30, 2, 0), dt(2023, 10, 29, 2, 0), dt(2024, 10, 27, 2, 0)]
    us_dst_start_dates = []
    us_dst_end_dates = []
    eu_dst_start_dates = []
    eu_dst_end_dates = []
    for year in years:
        us_dst_start_dates.append(get_us_dst_start(year))
        us_dst_end_dates.append(get_us_dst_end(year))
        eu_dst_start_dates.append(get_eu_dst_start(year))
        eu_dst_end_dates.append(get_eu_dst_end(year))
    assert us_dst_start_dates == expected_us_dst_start_dates
    assert us_dst_end_dates == expected_us_dst_end_dates
    assert eu_dst_start_dates == expected_eu_dst_start_dates
    assert eu_dst_end_dates == expected_eu_dst_end_dates


def test_build_time_features_df():
    date_list = pd.date_range(start=dt(2019, 1, 1), periods=24 * 365, freq="H").tolist()

    df0 = pd.DataFrame({"ts": date_list})
    time_df = build_time_features_df(dt=df0["ts"], conti_year_origin=2019)
    assert time_df["datetime"][0] == datetime.datetime(2019, 1, 1, 0, 0, 0)
    assert time_df["date"][0] == datetime.date(2019, 1, 1)
    assert time_df["year"][0] == 2019
    assert time_df["year_length"][0] == 365
    assert time_df["quarter"][0] == 1
    assert time_df["quarter_start"][0] == pd.to_datetime("2019-01-01")
    assert time_df["quarter_start"][24 * 89] == pd.to_datetime("2019-01-01")
    assert time_df["quarter_start"][24 * 91] == pd.to_datetime("2019-04-01")
    assert time_df["toq"][0] == 0
    assert time_df["toq"][24 * 16] == 16.0 / 90.0
    assert time_df["toq"][24 * 10] == 10.0 / 90.0
    assert time_df["toq"][24 * 89] == 89.0 / 90.0
    assert time_df["toq"][24 * 91] == 1.0 / 91.0
    assert time_df["month"][0] == 1
    assert time_df["month_length"][0] == 31
    assert time_df["woy"][0] == 1
    assert time_df["doy"][0] == 1
    assert time_df["dom"][0] == 1
    assert time_df["dow"][0] == 2
    assert time_df["str_dow"][0] == "2-Tue"
    assert time_df["hour"][0] == 0
    assert time_df["minute"][0] == 0
    assert time_df["second"][0] == 0
    assert time_df["year_quarter"][24 * 80] == "2019-1"
    assert time_df["year_month"][0] == "2019-01"
    assert time_df["year_woy"][0] == "2019_01"
    assert time_df["month_dom"][0] == "01/01"
    assert time_df["year_woy_dow"][0] == "2019_01_2"
    assert time_df["dow_hr"][0] == "2_00"
    assert time_df["dow_hr_min"][0] == "2_00_00"
    assert time_df["year_iso"].iloc[-1] == 2020
    assert time_df["year_woy_iso"].iloc[-1] == "2020_01"
    assert time_df["year_woy_dow_iso"].iloc[-1] == "2020_01_2"
    assert time_df["tod"][0] == 0.0
    assert time_df["tow"][0] == 1.0
    assert time_df["tom"][0] == 0.0 / 31
    assert time_df["toy"][0] == 0.0
    assert time_df["conti_year"][0] == 2019.0
    assert not time_df["is_weekend"][0]
    assert time_df["dow_grouped"][0] == "1234-MTuWTh"
    assert time_df["dow_grouped"][24 * 3] == "5-Fri"
    assert time_df["dow_grouped"][24 * 4] == "6-Sat"
    assert time_df["dow_grouped"][24 * 5] == "7-Sun"
    # detailed check on dow_hr
    assert list(time_df["dow_hr"])[::7][:25] == [
        "2_00",
        "2_07",
        "2_14",
        "2_21",
        "3_04",
        "3_11",
        "3_18",
        "4_01",
        "4_08",
        "4_15",
        "4_22",
        "5_05",
        "5_12",
        "5_19",
        "6_02",
        "6_09",
        "6_16",
        "6_23",
        "7_06",
        "7_13",
        "7_20",
        "1_03",
        "1_10",
        "1_17",
        "2_00",
    ]  # noqa: E501

    assert time_df["ct1"][0] == 0.0
    assert time_df["ct2"][0] == 0.0
    assert time_df["ct3"][0] == 0.0
    assert time_df["ct_sqrt"][0] == 0.0
    assert time_df["ct_root3"][0] == 0.0

    ct1 = 50.0 / 365 / 24
    assert time_df["ct1"][50] == pytest.approx(ct1, rel=1e-3)
    assert time_df["ct2"][50] == pytest.approx(ct1**2, rel=1e-3)
    assert time_df["ct3"][50] == pytest.approx(ct1**3, rel=1e-3)
    assert time_df["ct_sqrt"][50] == pytest.approx(ct1**0.5, rel=1e-3)
    assert time_df["ct_root3"][50] == pytest.approx(ct1 ** (1 / 3), rel=1e-3)

    quarter_dates = [
        "2020-01-01",
        "2020-03-31",  # Q1 2020 (leap year)
        "2020-04-01",
        "2020-06-30",  # Q2 2020
        "2020-07-01",
        "2020-09-30",  # Q3 2020
        "2020-10-01",
        "2020-12-31",  # Q4 2020
        "2021-01-01",
        "2021-03-31",  # Q1 2021
        "2021-05-13-12",
        "2021-08-03-18",  # Q2/3 2021
    ]
    time_df = build_time_features_df(quarter_dates, conti_year_origin=2020.0)
    assert_equal(
        time_df["quarter_start"],
        pd.Series(
            pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-01",
                    "2020-04-01",
                    "2020-04-01",
                    "2020-07-01",
                    "2020-07-01",
                    "2020-10-01",
                    "2020-10-01",
                    "2021-01-01",
                    "2021-01-01",
                    "2021-04-01",
                    "2021-07-01",
                ]
            )
        ),
        check_names=False,
    )
    assert_equal(
        time_df["quarter_length"],
        pd.Series(
            [
                91,
                91,
                91,
                91,
                92,
                92,
                92,
                92,
                90,
                90,
                91,
                92,
            ]
        ),
        check_names=False,
    )
    assert_equal(
        time_df["doq"],
        pd.Series(
            [
                1,
                91,
                1,
                91,
                1,
                92,
                1,
                92,
                1,
                90,
                43,
                34,
            ]
        ),
        check_names=False,
    )
    assert_equal(
        time_df["toq"],
        pd.Series(
            [
                0.0,
                90.0 / 91.0,
                0.0,
                90.0 / 91.0,
                0.0,
                91.0 / 92.0,
                0.0,
                91.0 / 92.0,
                0.0,
                89.0 / 90.0,
                42.5 / 91.0,
                33.75 / 92.0,
            ]
        ),
        check_names=False,
    )

    # Checks for exception
    with pytest.raises(ValueError, match="Length of dt cannot be zero."):
        build_time_features_df(dt=df0.iloc[0:0]["ts"], conti_year_origin=2019)


def test_build_time_features_df_with_dst():
    date_list = pd.date_range(start=dt(2022, 3, 13), periods=5, freq="H").tolist()

    df0 = pd.DataFrame({"ts": date_list})
    time_df = build_time_features_df(dt=df0["ts"], conti_year_origin=2022, add_dst_info=True)

    assert (time_df["us_dst"] == [False, False, True, True, True]).all()
    assert (time_df["eu_dst"] == [False] * 5).all()


def test_build_time_features_df_leap_years():
    date_list_non_leap_year = pd.date_range(start=dt(2019, 2, 28), periods=3 * 24, freq="H").tolist()

    df0 = pd.DataFrame({"ts": date_list_non_leap_year})
    time_df = build_time_features_df(dt=df0["ts"], conti_year_origin=2019)
    expected = np.repeat([58.0, 59.0, 60.0], 24) + np.tile(range(24), 3) / 24
    observed = 365.0 * time_df["toy"]
    assert np.allclose(observed, expected)

    date_list_leap_year = pd.date_range(start=dt(2020, 2, 28), periods=3 * 24, freq="H").tolist()

    df0 = pd.DataFrame({"ts": date_list_leap_year})
    time_df = build_time_features_df(dt=df0["ts"], conti_year_origin=2019)
    expected = np.repeat([58.0, 59.0, 59.0], 24) + np.concatenate([range(24), np.repeat(0, 24), range(24)]) / 24
    observed = 365.0 * time_df["toy"]
    assert np.allclose(observed, expected)


def test_build_time_features_df_arabic_persian():
    """Tests Hijri and Persian calendar features in build_time_features_df."""
    import pandas as pd
    from abvelocity.ts.common.features.timeseries_features import build_time_features_df, gregorian_to_hijri, gregorian_to_persian

    # ── Hijri: 1 Ramadan 1445 = 2024-03-11 ──────────────────────────────────
    dates = pd.date_range("2024-03-11", periods=5, freq="D")
    df = build_time_features_df(dates, conti_year_origin=2020)

    assert list(df["arabic_year"]) == [1445] * 5
    assert list(df["arabic_month"]) == [9] * 5  # Ramadan = month 9
    assert list(df["arabic_dom"]) == [1, 2, 3, 4, 5]
    # arabic_toy: 1 Ramadan = (9-1)/12 + (1-1)/354 = 0.6667
    assert abs(df["arabic_toy"].iloc[0] - 8 / 12.0) < 1e-6
    # toy increments by 1/354 per day
    assert abs(df["arabic_toy"].iloc[1] - df["arabic_toy"].iloc[0] - 1 / 354.0) < 1e-6

    # ── Hijri: Eid al-Adha 1445 = 2024-06-17 (10 Dhu al-Hijja) ─────────────
    hijri_year, hijri_month, hijri_day = gregorian_to_hijri(np.array([2024]), np.array([6]), np.array([17]))
    assert hijri_year[0] == 1445
    assert hijri_month[0] == 12  # Dhu al-Hijja
    assert hijri_day[0] == 10

    # ── Hijri epoch: 1 Muharram 1 AH = 622-07-19 Gregorian proleptic ────────
    hijri_year, hijri_month, hijri_day = gregorian_to_hijri(np.array([622]), np.array([7]), np.array([19]))
    assert (hijri_year[0], hijri_month[0], hijri_day[0]) == (1, 1, 1)

    # ── Persian: last days of 1402 then Nowruz 1403 ─────────────────────────
    dates_nowruz = pd.date_range("2024-03-19", periods=4, freq="D")
    df_nowruz = build_time_features_df(dates_nowruz, conti_year_origin=2020)

    # 2024-03-19 still in 1402; 2024-03-20 = Nowruz 1403 (1 Farvardin)
    assert df_nowruz["persian_year"].iloc[0] == 1402
    assert df_nowruz["persian_year"].iloc[1] == 1403
    assert df_nowruz["persian_month"].iloc[1] == 1  # Farvardin
    assert df_nowruz["persian_dom"].iloc[1] == 1

    # ── Persian month structure: months 1-6 = 31 days ───────────────────────
    dates_tir = pd.date_range("2024-06-21", periods=3, freq="D")  # 1-3 Tir 1403
    df_tir = build_time_features_df(dates_tir, conti_year_origin=2020)
    assert list(df_tir["persian_month"]) == [4, 4, 4]  # Tir = month 4
    assert list(df_tir["persian_dom"]) == [1, 2, 3]

    # ── Persian: Nowruz always on March 20 or 21 (sampled years) ────────────
    for greg_year in [2000, 2010, 2020, 2024, 2025, 2030]:
        mar20 = gregorian_to_persian(np.array([greg_year]), np.array([3]), np.array([20]))
        mar21 = gregorian_to_persian(np.array([greg_year]), np.array([3]), np.array([21]))
        nowruz_on_20 = mar20[1][0] == 1 and mar20[2][0] == 1
        nowruz_on_21 = mar21[1][0] == 1 and mar21[2][0] == 1
        assert nowruz_on_20 or nowruz_on_21, f"Nowruz not on Mar 20/21 for {greg_year}"

    # ── Both calendars: new columns present in TimeFeaturesEnum ─────────────
    from abvelocity.ts.common.constants import TimeFeaturesEnum

    for col in ["arabic_year", "arabic_month", "arabic_dom", "arabic_toy", "persian_year", "persian_month", "persian_dom", "persian_toy"]:
        assert col in TimeFeaturesEnum.__dict__["_member_names_"], f"{col} missing from TimeFeaturesEnum"
        assert col in df.columns, f"{col} missing from build_time_features_df output"


def test_add_time_features_df():
    """Tests add_time_features_df"""
    # create indexed input
    date_list = pd.date_range(start=datetime.datetime(2019, 1, 1), periods=100, freq="H").tolist()
    df0 = pd.DataFrame({TIME_COL: date_list}, index=date_list)

    df = add_time_features_df(df=df0, time_col=TIME_COL, conti_year_origin=2018)
    assert df["year"][0] == 2019
    assert df.shape[0] == df0.shape[0]

    hourly_data = generate_df_with_reg_for_tests(freq="H", periods=24 * 500, train_start_date=datetime.datetime(2018, 7, 1), conti_year_origin=2018)
    cols = [TIME_COL, "regressor1", "regressor_bool", "regressor_categ"]
    train_df = hourly_data["train_df"]
    df = add_time_features_df(df=train_df[cols], time_col=TIME_COL, conti_year_origin=2018)
    assert df["year"][0] == 2018
    assert (df["dow_hr"][:3] == ["7_00", "7_01", "7_02"]).all()
    assert df.shape[0] == train_df.shape[0]


def test_get_holidays():
    """Tests get_holidays"""
    # request holidays by country code
    countries = ["CN", "IN", "US", "UK"]
    res_code = get_holidays(countries, year_start=2017, year_end=2025)

    in_df = res_code["IN"]
    row_index = in_df[EVENT_DF_DATE_COL] == "2017-01-26"
    assert in_df.loc[row_index, EVENT_DF_LABEL_COL].values[0] == "Republic"
    row_index = in_df[EVENT_DF_DATE_COL] == "2017-08-15"
    assert "Independence" in in_df.loc[row_index, EVENT_DF_LABEL_COL].values[0]
    uk_df = res_code["UK"]
    row_index = uk_df[EVENT_DF_DATE_COL] == "2024-03-29"
    assert uk_df.loc[row_index, EVENT_DF_LABEL_COL].values[0] == "Good Friday"
    us_df = res_code["US"]
    row_index = us_df[EVENT_DF_DATE_COL] == "2017-01-16"
    assert us_df.loc[row_index, EVENT_DF_LABEL_COL].values[0] == "MLK"
    row_index = us_df[EVENT_DF_DATE_COL] == "2017-10-31"
    assert us_df.loc[row_index, EVENT_DF_LABEL_COL].values[0] == "Halloween"
    cn_df = res_code["CN"]
    row_index = cn_df[EVENT_DF_DATE_COL] == "2017-01-28"
    assert cn_df.loc[row_index, EVENT_DF_LABEL_COL].values[0] == "Lunar New Year"
    row_index = cn_df[EVENT_DF_DATE_COL] == "2025-05-31"
    assert cn_df.loc[row_index, EVENT_DF_LABEL_COL].values[0] == "Dragon Boat Festival"

    # request holidays by full country name
    countries = ["China", "India", "UnitedStates", "UnitedKingdom"]
    res_full = get_holidays(countries, year_start=2017, year_end=2025)
    names = [
        ("CN", "China"),
        ("IN", "India"),
        ("US", "UnitedStates"),
        ("UK", "UnitedKingdom"),
    ]
    for code_name, full_name in names:
        assert res_code[code_name].equals(res_full[full_name])

    for country, holidays in res_full.items():
        assert holidays[EVENT_DF_LABEL_COL].str.contains("/").sum() == 0, "Holiday names cannot contain '/'!"

    with pytest.raises(AttributeError, match="Holidays in unknown country are not currently supported!"):
        get_holidays(["unknown country"], year_start=2017, year_end=2025)


def test_get_available_holiday_lookup_countries():
    """Tests get_available_holiday_lookup_countries"""
    valid_countries = get_available_holiday_lookup_countries()
    assert "Croatia" in valid_countries
    assert "UnitedStates" in valid_countries
    assert "Indonesia" in valid_countries
    assert "datetime" not in valid_countries
    assert len(valid_countries) == 774

    countries = ["IN", "India", "US", "UnitedStates", "UK"]
    valid_countries = get_available_holiday_lookup_countries(countries)
    assert len(valid_countries) == 5

    countries = ["United States", "SomeOtherPlace"]
    valid_countries = get_available_holiday_lookup_countries(countries)
    assert len(valid_countries) == 0


def test_get_available_holidays_in_countries():
    """Tests get_available_holidays_in_countries"""
    available_holidays = get_available_holidays_in_countries(countries=["CN", "US"], year_start=2017, year_end=2025)
    # CN: verify key holidays are present (names after sanitize_name + normalize_name pipeline)
    cn_holidays = available_holidays["CN"]
    assert "Lunar New Year" in cn_holidays  # raw: "Lunar New Year" → canonical
    assert "Dragon Boat Festival" in cn_holidays
    assert "Labor" in cn_holidays  # raw: "Labor"
    assert "Mid-Autumn Festival" in cn_holidays
    assert "National Day" in cn_holidays  # "National" is in KEEP_DAY_SUFFIX — Day preserved
    assert "New Years" in cn_holidays  # raw: "New Years"
    assert "Tomb-Sweeping" in cn_holidays  # raw: "Tomb-Sweeping Day"
    # US: base holidays + _observed variants kept as distinct entries
    assert available_holidays["US"] == [
        "Christmas",
        "Christmas_observed",
        "Columbus",
        "Halloween",
        "Independence",
        "Independence_observed",
        "Juneteenth National Independence",
        "Juneteenth National Independence_observed",
        "Labor",
        "MLK",
        "Memorial",
        "New Years",
        "New Years_observed",
        "Thanksgiving",
        "Veterans",
        "Veterans_observed",
        "Washingtons Birthday",
    ]


def test_get_available_holidays_across_countries():
    """Tests get_available_holidays_across_countries"""
    available_holidays = get_available_holidays_across_countries(countries=["CN", "US"], year_start=2017, year_end=2025)
    # Check key holidays from both countries are present (sanitized names)
    assert "Lunar New Year" in available_holidays  # raw: "Lunar New Year" → canonical
    assert "Dragon Boat Festival" in available_holidays
    assert "Halloween" in available_holidays
    assert "Thanksgiving" in available_holidays  # raw: "Thanksgiving"
    assert "MLK" in available_holidays  # raw: "MLK"
    assert "Tomb-Sweeping" in available_holidays  # raw: "Tomb-Sweeping Day"
    assert "National Day" in available_holidays  # KEEP_DAY_SUFFIX exception
    # All US holidays should be present
    us_holidays = get_available_holidays_in_countries(["US"], year_start=2017, year_end=2025)["US"]
    for h in us_holidays:
        assert h in available_holidays


def test_add_daily_events():
    """Tests ``add_daily_events`` function."""
    # Generates events dictionary
    countries = ["US", "India", "UK"]
    event_df_dict = get_holidays(countries, year_start=2015, year_end=2025)
    original_col_names = [event_df_dict[country].columns[1] for country in countries]

    # Generates temporal data
    date_list = pd.date_range(start=dt(2019, 1, 1), periods=100, freq="H").tolist()

    df0 = pd.DataFrame({"ts": date_list})
    df = add_time_features_df(df0, time_col="ts", conti_year_origin=2018)
    df_with_events = add_daily_events(df=df, event_df_dict=event_df_dict, date_col="date")

    assert df_with_events[f"{EVENT_PREFIX}_US"].values[0] == "New Years"
    assert df_with_events[f"{EVENT_PREFIX}_India"].values[0] == ""  # India doesn't observe New Years nationally
    assert df_with_events[f"{EVENT_PREFIX}_US"].values[25] == ""

    # Makes sure the function does not modify the input
    new_col_names = [event_df_dict[country].columns[1] for country in countries]
    assert original_col_names == new_col_names

    # Tests event indicators are correctly included
    expected_new_cols = [f"events_{country}" for country in countries] + [IS_EVENT_EXACT_COL, IS_EVENT_ADJACENT_COL, IS_EVENT_COL]  # 6 columns should be added
    assert set(df_with_events.columns).difference(df.columns) == set(expected_new_cols)
    num_is_event = df_with_events[IS_EVENT_COL].sum()
    num_is_event_exact = df_with_events[IS_EVENT_EXACT_COL].sum()
    num_is_event_adjacent = df_with_events[IS_EVENT_ADJACENT_COL].sum()
    assert num_is_event == num_is_event_exact
    assert num_is_event_exact > 0
    assert num_is_event_adjacent == 0
    assert num_is_event == num_is_event_exact + num_is_event_adjacent


def test_add_daily_events_with_neighbor_impact():
    """Tests adding daily events with neighbor impact."""
    # Tests weekly data.
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", freq="W-SUN", periods=100), "y": 0})
    countries = ["US"]
    event_df_dict = get_holidays(countries, year_start=2015, year_end=2025)
    new_df = add_daily_events(df, event_df_dict, neighbor_impact=lambda x: [x - timedelta(days=x.isocalendar()[2] - 1) + timedelta(days=i) for i in range(7)])
    # Checks holidays are mapped to the correct weekly dates.
    assert new_df.iloc[0].tolist() == [pd.Timestamp("2020-01-05"), 0, "New Years", 1, 0, 1]
    assert new_df.iloc[-1].tolist() == [pd.Timestamp("2021-11-28"), 0, "Thanksgiving", 1, 0, 1]

    # Tests daily data, assuming rolling 7 day.
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", freq="D", periods=500), "y": 0})
    countries = ["US"]
    event_df_dict = get_holidays(countries, year_start=2015, year_end=2025)
    new_df = add_daily_events(df, event_df_dict, neighbor_impact=7)
    # Checks holidays are mapped to the correct daily dates.
    assert new_df.iloc[0].tolist() == [pd.Timestamp("2020-01-01"), 0, "Christmas", 1, 0, 1]
    assert new_df.iloc[1].tolist() == [pd.Timestamp("2020-01-01"), 0, "New Years", 1, 0, 1]
    assert new_df.iloc[2].tolist() == [pd.Timestamp("2020-01-02"), 0, "New Years", 1, 0, 1]

    # Tests daily data, assuming a list of customized neighboring effect.
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", freq="D", periods=500), "y": 0})
    countries = ["US"]
    event_df_dict = get_holidays(countries, year_start=2015, year_end=2025)
    new_df = add_daily_events(df, event_df_dict, neighbor_impact=[1, 4])
    # Checks holidays are mapped to the correct daily dates.
    assert new_df.iloc[0].tolist() == [pd.Timestamp("2020-01-01"), 0, "New Years", 1, 0, 1]
    assert new_df.iloc[1].tolist() == [pd.Timestamp("2020-01-02"), 0, "New Years", 1, 0, 1]
    assert new_df.iloc[4].tolist() == [pd.Timestamp("2020-01-05"), 0, "New Years", 1, 0, 1]


def test_add_daily_event_shifted_effect():
    """Tests adding additional neighbor events.
    The additional events are added as extra columns rather than extra dates
    under the same columns as in ``neighbor_effect``.
    """
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", freq="W-SUN", periods=100), "y": 0})
    countries = ["US"]
    event_df_dict = get_holidays(countries, year_start=2015, year_end=2025)
    new_df = add_daily_events(
        df,
        event_df_dict,
        neighbor_impact=lambda x: [x - timedelta(days=x.isocalendar()[2] - 1) + timedelta(days=i) for i in range(7)],
        shifted_effect=["-7D", "7D"],
    )
    assert new_df.iloc[0].tolist() == [pd.Timestamp("2020-01-05"), 0, "New Years", "", "Christmas_7D_after", 1, 0, 1]
    assert new_df.iloc[1].tolist() == [pd.Timestamp("2020-01-12"), 0, "", "", "New Years_7D_after", 1, 0, 1]
    assert new_df.iloc[2].tolist() == [pd.Timestamp("2020-01-19"), 0, "", "MLK_7D_before", "", 1, 0, 1]
    assert new_df.iloc[-1].tolist() == [pd.Timestamp("2021-11-28"), 0, "Thanksgiving", "", "", 1, 0, 1]


def test_get_evenly_spaced_changepoints():
    df = pd.DataFrame({"time_col": np.arange(1, 11)})
    result = get_evenly_spaced_changepoints_values(df, "time_col", n_changepoints=3)
    assert np.all(result == np.array([3, 6, 8]))

    df = pd.DataFrame({"time_col": np.arange(-2, 8)})
    result = get_evenly_spaced_changepoints_values(df, "time_col", n_changepoints=2)
    assert np.all(result == np.array([1, 4]))

    with pytest.raises(ValueError, match="n_changepoints must be > 0"):
        get_evenly_spaced_changepoints_values(df, "time_col", n_changepoints=0)


def test_get_evenly_spaced_changepoints_dates():
    n_changepoint = 3
    df = pd.DataFrame({"ts": [dt(2020, 1, i) for i in range(1, 10)], "y": list(range(1, 10))})
    changepoints = get_evenly_spaced_changepoints_dates(n_changepoints=n_changepoint, df=df, time_col="ts")
    expected_changepoints = pd.Series([dt(2020, 1, 1), dt(2020, 1, 3), dt(2020, 1, 5), dt(2020, 1, 7)])
    assert changepoints.reset_index()["ts"].equals(expected_changepoints)
    # tests non-rangeIndex
    df.index = [3, 5, 7, 9, 11, 13, 15, 17, 19]
    changepoints = get_evenly_spaced_changepoints_dates(n_changepoints=n_changepoint, df=df, time_col="ts")
    assert changepoints.reset_index()["ts"].equals(expected_changepoints)
    # test n_changepoint = 0
    n_changepoint = 0
    changepoints = get_evenly_spaced_changepoints_dates(n_changepoints=n_changepoint, df=df, time_col="ts")
    expected_changepoints = pd.Series([dt(2020, 1, 1)])
    assert changepoints.reset_index()["ts"].equals(expected_changepoints)


def test_get_changepoint_string():
    # tests daily data
    changepoint_dates = ["2020-01-01", "2020-02-03"]
    changepoint_string = get_changepoint_string(changepoint_dates)
    expected_changepoint_string = ["_2020_01_01_00", "_2020_02_03_00"]
    assert changepoint_string == expected_changepoint_string
    # tests hourly data
    changepoint_dates = ["2020-01-01-05", "2020-02-03-04"]
    changepoint_string = get_changepoint_string(changepoint_dates)
    expected_changepoint_string = ["_2020_01_01_05", "_2020_02_03_04"]
    assert changepoint_string == expected_changepoint_string
    # tests minute data
    changepoint_dates = ["2020-01-01", "2020-02-03 04:05"]
    changepoint_string = get_changepoint_string(changepoint_dates)
    expected_changepoint_string = ["_2020_01_01_00_00", "_2020_02_03_04_05"]
    assert changepoint_string == expected_changepoint_string
    # tests second data
    changepoint_dates = ["2020-01-01", "2020-02-03 04:05:06"]
    changepoint_string = get_changepoint_string(changepoint_dates)
    expected_changepoint_string = ["_2020_01_01_00_00_00", "_2020_02_03_04_05_06"]
    assert changepoint_string == expected_changepoint_string


def test_get_evenly_spaced_changepoint_values():
    df = pd.DataFrame({"time_col": np.arange(1, 11), "ts": pd.date_range(start="2020-01-01", periods=10, freq="AS")})
    changepoints = get_evenly_spaced_changepoints_values(df, "time_col", n_changepoints=3)
    changepoint_dates = get_changepoint_dates_from_changepoints_dict(changepoints_dict={"method": "uniform", "n_changepoints": 3}, df=df, time_col="ts")

    # linear growth
    changepoint_df = get_changepoint_features(df, changepoints, continuous_time_col="time_col", growth_func=None, changepoint_dates=changepoint_dates)
    expected = pd.DataFrame(
        {
            "changepoint0_2022_01_01_00": [0, 0, 0, 1, 2, 3, 4, 5, 6, 7],
            "changepoint1_2025_01_01_00": [0, 0, 0, 0, 0, 0, 1, 2, 3, 4],
            "changepoint2_2027_01_01_00": [0, 0, 0, 0, 0, 0, 0, 0, 1, 2],
        }
    )
    assert changepoint_df.equals(expected)

    # quadratic growth
    changepoint_df = get_changepoint_features(df, changepoints, continuous_time_col="time_col", growth_func=lambda x: x**2, changepoint_dates=changepoint_dates)
    expected = pd.DataFrame(
        {
            "changepoint0_2022_01_01_00": [0, 0, 0, 1, 4, 9, 16, 25, 36, 49],
            "changepoint1_2025_01_01_00": [0, 0, 0, 0, 0, 0, 1, 4, 9, 16],
            "changepoint2_2027_01_01_00": [0, 0, 0, 0, 0, 0, 0, 0, 1, 4],
        }
    )
    assert changepoint_df.equals(expected)

    # real example
    n_changepoints = 3
    date_list = pd.date_range(start=dt(2019, 1, 1), periods=20, freq="H").tolist()

    df0 = pd.DataFrame({"ts": date_list})
    df = add_time_features_df(df0, time_col="ts", conti_year_origin=2018)
    changepoints = get_evenly_spaced_changepoints_values(df, "ct1", n_changepoints=n_changepoints)
    changepoint_dates = get_changepoint_dates_from_changepoints_dict(changepoints_dict={"method": "uniform", "n_changepoints": 3}, df=df, time_col="ts")
    changepoint_df = get_changepoint_features(df, changepoints, continuous_time_col="ct1", growth_func=lambda x: x, changepoint_dates=changepoint_dates)
    assert changepoint_df.shape == (df.shape[0], n_changepoints)
    assert changepoint_df.iloc[15, 2] == 0.0
    assert changepoint_df.iloc[16, 2] == 0.00011415525113989133


def test_get_custom_changepoints():
    """Tests get_custom_changepoints and get_changepoint_features"""
    date_list = pd.date_range(start=dt(2019, 1, 1), periods=20, freq="D").tolist()
    time_col = "custom_time_col"
    df0 = pd.DataFrame({time_col: date_list})
    df = add_time_features_df(df0, time_col="custom_time_col", conti_year_origin=2018)

    # dates as datetime
    changepoint_dates = pd.to_datetime(["2018-01-01", "2019-01-02-16", "2019-01-03", "2019-02-01"], format="mixed")
    result = get_custom_changepoints_values(
        df=df, changepoint_dates=changepoint_dates, time_col=time_col, continuous_time_col=time_col  # pd.Timestamp type  # makes checking the result easier
    )
    # 2018-01-01 is mapped to 2019-01-01. Duplicates mapped to "2019-01-03" are merged
    # Last requested changepoint is not found
    assert np.all(result == pd.to_datetime(["2019-01-01", "2019-01-03"]))

    # dates as strings
    changepoint_dates = ["2018-01-01", "2019-01-02-16", "2019-01-03", "2019-02-01"]
    result = get_custom_changepoints_values(df=df, changepoint_dates=changepoint_dates, time_col="date", continuous_time_col=time_col)  # datetime.date type
    assert np.all(result == pd.to_datetime(["2019-01-01", "2019-01-03"]))

    # continuous_time_col different from time_col
    # check using timestamps from last `result`
    changepoints = get_custom_changepoints_values(df=df, changepoint_dates=changepoint_dates, time_col=time_col, continuous_time_col="ct1")
    assert np.all(changepoints == df[df[time_col].isin(result)]["ct1"].values)

    # generated features, using changepoints from above
    changepoint_df = get_changepoint_features(
        df,
        changepoints,
        continuous_time_col="ct1",
        growth_func=lambda x: x,
    )
    assert changepoint_df.shape == (df.shape[0], len(result))
    assert round(changepoint_df.iloc[15, 0], 3) == 0.041
    assert round(changepoint_df.iloc[16, 1], 3) == 0.038

    # no matching dates
    changepoint_dates = ["2019-02-01"]
    result = get_custom_changepoints_values(df=df, changepoint_dates=changepoint_dates, time_col=time_col, continuous_time_col=time_col)
    assert result is None

    # hourly data, provided changepoints at daily level
    date_list = pd.date_range(start=dt(2019, 1, 1), periods=20 * 24, freq="H").tolist()
    time_col = "custom_time_col"
    df0 = pd.DataFrame({time_col: date_list})
    df = add_time_features_df(df0, time_col="custom_time_col", conti_year_origin=2018)

    # dates as datetime
    changepoint_dates = pd.to_datetime(["2018-01-01", "2019-01-02-16", "2019-01-03", "2019-02-01"], format="mixed")
    result = get_custom_changepoints_values(
        df=df, changepoint_dates=changepoint_dates, time_col=time_col, continuous_time_col=time_col  # pd.Timestamp type  # makes checking the result easier
    )
    # 2018-01-01 is mapped to 2019-01-01-00. Mapped to -00 if no hour provided
    # Last requested changepoint is not found
    assert np.all(result == pd.to_datetime(["2019-01-01-00", "2019-01-02-16", "2019-01-03-00"], format="mixed"))


def test_get_changepoint_values_from_config(hourly_data):
    """Tests get_changepoint_values_from_config"""
    train_df = hourly_data["train_df"]
    conti_year_origin = get_default_origin_for_time_vars(train_df, TIME_COL)
    time_features_df = build_time_features_df(dt=train_df[TIME_COL], conti_year_origin=conti_year_origin)
    with pytest.raises(Exception, match="changepoint method must be specified"):
        get_changepoint_values_from_config(changepoints_dict={"n_changepoints": 2}, time_features_df=time_features_df, time_col="datetime")

    with pytest.raises(NotImplementedError, match="changepoint method.*not recognized"):
        get_changepoint_values_from_config(changepoints_dict={"method": "not implemented"}, time_features_df=time_features_df, time_col="datetime")

    # tests uniform method
    changepoint_values = get_changepoint_values_from_config(
        changepoints_dict={"method": "uniform", "n_changepoints": 20}, time_features_df=time_features_df, time_col="datetime"
    )
    expected_changepoint_values = get_evenly_spaced_changepoints_values(df=time_features_df, n_changepoints=20)
    assert np.array_equal(changepoint_values, expected_changepoint_values)

    changepoint_values = get_changepoint_values_from_config(
        changepoints_dict={"method": "uniform", "n_changepoints": 20, "continuous_time_col": "ct2"}, time_features_df=time_features_df, time_col="datetime"
    )
    expected_changepoint_values = get_evenly_spaced_changepoints_values(df=time_features_df, n_changepoints=20, continuous_time_col="ct2")
    assert np.array_equal(changepoint_values, expected_changepoint_values)

    # tests custom method
    dates = ["2018-01-01", "2019-01-02-16", "2019-01-03", "2019-02-01"]
    changepoint_values = get_changepoint_values_from_config(
        changepoints_dict={"method": "custom", "dates": dates}, time_features_df=time_features_df, time_col="datetime"
    )
    expected_changepoint_values = get_custom_changepoints_values(df=time_features_df, changepoint_dates=dates, time_col="datetime")
    assert np.array_equal(changepoint_values, expected_changepoint_values)

    changepoint_values = get_changepoint_values_from_config(
        changepoints_dict={"method": "custom", "dates": dates, "continuous_time_col": "ct2"}, time_features_df=time_features_df, time_col="datetime"
    )
    expected_changepoint_values = get_custom_changepoints_values(df=time_features_df, changepoint_dates=dates, time_col="datetime", continuous_time_col="ct2")
    assert np.array_equal(changepoint_values, expected_changepoint_values)


def test_get_changepoint_features_and_values_from_config(hourly_data):
    """Tests get_changepoint_features_and_values_from_config"""
    # no changepoints
    train_df = hourly_data["train_df"]
    changepoints = get_changepoint_features_and_values_from_config(df=train_df, time_col=TIME_COL, changepoints_dict=None, origin_for_time_vars=None)
    assert changepoints["changepoint_df"] is None
    assert changepoints["changepoint_values"] is None
    assert changepoints["continuous_time_col"] is None
    assert changepoints["growth_func"] is None
    assert changepoints["changepoint_cols"] == []

    # uniform method
    train_df = hourly_data["train_df"]
    n_changepoints = 20
    changepoints_dict = {"method": "uniform", "n_changepoints": n_changepoints, "continuous_time_col": "ct2"}
    changepoints = get_changepoint_features_and_values_from_config(
        df=train_df, time_col=TIME_COL, changepoints_dict=changepoints_dict, origin_for_time_vars=None
    )
    changepoint_dates = get_changepoint_dates_from_changepoints_dict(changepoints_dict=changepoints_dict, df=train_df, time_col=TIME_COL)

    assert sorted(list(changepoints.keys())) == sorted(["changepoint_df", "changepoint_values", "continuous_time_col", "growth_func", "changepoint_cols"])
    assert changepoints["changepoint_df"].shape == (train_df.shape[0], n_changepoints)
    assert changepoints["changepoint_values"].shape == (n_changepoints,)
    assert changepoints["continuous_time_col"] == "ct2"
    assert changepoints["growth_func"] is None
    assert changepoints["changepoint_cols"] == [
        f"{CHANGEPOINT_COL_PREFIX}{i}_{pd.to_datetime(date).strftime('%Y_%m_%d_%H')}" for i, date in enumerate(changepoint_dates)
    ]

    # custom method, no changepoint in range
    changepoints_dict = {"method": "custom", "dates": ["2048-03-02"], "growth_func": signed_sqrt}
    origin_for_time_vars = convert_date_to_continuous_time(datetime.datetime(2018, 1, 1))
    changepoints = get_changepoint_features_and_values_from_config(
        df=train_df, time_col=TIME_COL, changepoints_dict=changepoints_dict, origin_for_time_vars=origin_for_time_vars
    )

    assert sorted(list(changepoints.keys())) == sorted(["changepoint_df", "changepoint_values", "continuous_time_col", "growth_func", "changepoint_cols"])
    assert changepoints["changepoint_df"] is None
    assert changepoints["changepoint_values"] is None
    assert changepoints["continuous_time_col"] is None
    assert changepoints["growth_func"] == signed_sqrt
    assert changepoints["changepoint_cols"] == []


def test_get_changepoint_dates_from_changepoints_dict():
    # tests None
    changepoint_dates = get_changepoint_dates_from_changepoints_dict(changepoints_dict=None)
    assert changepoint_dates is None
    changepoint_dates = get_changepoint_dates_from_changepoints_dict(changepoints_dict={"n_changepoints": 2})
    assert changepoint_dates is None
    changepoint_dates = get_changepoint_dates_from_changepoints_dict(changepoints_dict={"method": "other_key", "n_changepoints": 2})
    assert changepoint_dates is None
    # tests custom method
    changepoints_dict = {"method": "custom", "dates": [dt(2020, 1, 5), dt(2020, 5, 6)]}
    changepoint_dates = get_changepoint_dates_from_changepoints_dict(changepoints_dict=changepoints_dict)
    assert all(changepoint_dates == pd.to_datetime([dt(2020, 1, 5), dt(2020, 5, 6)]))
    # tests custom method with other type of dates
    changepoints_dict = {"method": "custom", "dates": np.array([dt(2020, 1, 5), dt(2020, 5, 6)])}
    changepoint_dates = get_changepoint_dates_from_changepoints_dict(changepoints_dict=changepoints_dict)
    assert all(changepoint_dates == pd.to_datetime([dt(2020, 1, 5), dt(2020, 5, 6)]))
    # tests uniform method
    df = pd.DataFrame({"ts": pd.date_range(start="2020-01-01", end="2020-01-10")})
    changepoints_dict = {"method": "uniform", "n_changepoints": 2}
    changepoint_dates = get_changepoint_dates_from_changepoints_dict(changepoints_dict=changepoints_dict, df=df, time_col="ts")
    assert all(changepoint_dates == pd.to_datetime([dt(2020, 1, 4), dt(2020, 1, 7)]))
    # tests uniform error
    with pytest.raises(ValueError, match="When the method of ``changepoints_dict`` is 'uniform', ``df`` and " "``time_col`` must be provided."):
        get_changepoint_dates_from_changepoints_dict(changepoints_dict=changepoints_dict)
    # tests auto error
    changepoints_dict = {"method": "auto"}
    with pytest.raises(ValueError, match="The method of ``changepoints_dict`` can not be 'auto'. " "Please specify or detect change points first."):
        get_changepoint_dates_from_changepoints_dict(changepoints_dict=changepoints_dict)


def test_add_event_window():
    """Tests add_event_window"""
    # generate events data
    event_df_dict = get_holidays(countries=["US", "India", "UK"], year_start=2018, year_end=2019)
    df = event_df_dict["US"]

    shifted_event_dict = add_event_window(
        df=df, time_col=EVENT_DF_DATE_COL, label_col=EVENT_DF_LABEL_COL, time_delta="1D", pre_num=2, post_num=1, events_name="US"
    )

    assert len(shifted_event_dict) == 3

    df = shifted_event_dict["US_minus_1"]
    assert df.loc[df[EVENT_DF_DATE_COL] == "2017-12-31"][EVENT_DF_LABEL_COL].values[0] == "New Years"

    df = shifted_event_dict["US_minus_2"]
    assert df.loc[df[EVENT_DF_DATE_COL] == "2017-12-30"][EVENT_DF_LABEL_COL].values[0] == "New Years"

    # With no shifts
    shifted_event_dict = add_event_window(
        df=df, time_col=EVENT_DF_DATE_COL, label_col=EVENT_DF_LABEL_COL, time_delta="1D", pre_num=0, post_num=0, events_name="US"
    )
    assert len(shifted_event_dict) == 0


def test_add_event_window_multi():
    # generating events data
    event_df_dict = get_holidays(countries=["US", "India", "UK"], year_start=2018, year_end=2019)

    shifted_event_dict = add_event_window_multi(
        event_df_dict=event_df_dict,
        time_col=EVENT_DF_DATE_COL,
        label_col=EVENT_DF_LABEL_COL,
        time_delta="1D",
        pre_num=1,
        post_num=1,
        pre_post_num_dict={"US": [2, 1], "India": [1, 1]},
    )

    # we expect the dict size below to be 7 as US will generate 3 = 2 + 1; Each of India and UK will generate 2 = 1 + 1
    assert len(shifted_event_dict) == 7

    df = shifted_event_dict["US_minus_1"]
    assert df.loc[df[EVENT_DF_DATE_COL] == "2017-12-31"][EVENT_DF_LABEL_COL].values[0] == "New Years"

    df = shifted_event_dict["US_minus_2"]
    assert df.loc[df[EVENT_DF_DATE_COL] == "2017-12-30"][EVENT_DF_LABEL_COL].values[0] == "New Years"

    df = shifted_event_dict["India_plus_1"]
    assert df.loc[df[EVENT_DF_DATE_COL] == "2018-01-27"][EVENT_DF_LABEL_COL].values[0] == "Republic"

    df = shifted_event_dict["UK_minus_1"]
    assert df.loc[df[EVENT_DF_DATE_COL] == "2017-12-31"][EVENT_DF_LABEL_COL].values[0] == "New Years"

    # With `pre_post_num_dict=None`
    shifted_event_dict = add_event_window_multi(
        event_df_dict=event_df_dict, time_col=EVENT_DF_DATE_COL, label_col=EVENT_DF_LABEL_COL, time_delta="1D", pre_num=1, post_num=1, pre_post_num_dict=None
    )
    assert len(shifted_event_dict) == 6


def test_get_fourier_col_name():
    assert get_fourier_col_name(2, "dow", function_name="sin") == "sin2_dow"
    assert get_fourier_col_name(4.0, "dow", function_name="cos") == "cos4_dow"
    assert get_fourier_col_name(2, "dow", function_name="sin", seas_name="weekly") == "sin2_dow_weekly"
    assert get_fourier_col_name(4.0, "dow", function_name="cos", seas_name="weekly") == "cos4_dow_weekly"


def test_fourier_series_fcn():
    x = np.linspace(2.0, 3.0, num=100)
    df1 = pd.DataFrame({"x": x})
    func = fourier_series_fcn(col_name="x", period=1.0, order=2)
    df2 = func(df1)["df"]
    """
    # visualization for debugging
    col1 = get_fourier_col_name(1, "x", function_name="sin")
    col2 = get_fourier_col_name(1, "x", function_name="cos")
    col3 = get_fourier_col_name(2, "x", function_name="sin")
    col4 = get_fourier_col_name(2, "x", function_name="cos")
    plt.plot(x, df2[col1], label=col1)
    plt.plot(x, df2[col2], label=col2)
    plt.plot(x, df2[col3], label=col3)
    plt.plot(x, df2[col4], label=col4)
    plt.legend()
    """
    assert df2[get_fourier_col_name(1, "x", function_name="sin")][0].round(1) == 0.0


def test_fourier_series_multi_fcn():
    """Tests fourier_series_multi_fcn"""
    x = np.linspace(2.0, 3.0, num=100)
    y = np.linspace(3.0, 4.0, num=100)
    df0 = pd.DataFrame({"x": x, "y": y})

    func = fourier_series_multi_fcn(
        col_names=["x", "x", "y"], periods=[1.0, 2.0, 0.5], orders=[3, 4, 2], seas_names=["cat_period", "double_cat_period", "dog_period"]
    )

    res = func(df0)
    df = res["df"]

    """
    col1 = get_fourier_col_name(1, "x", function_name="cos", seas_name="cat_period")
    col2 = get_fourier_col_name(3, "x", function_name="sin", seas_name="cat_period")
    plt.plot(df["x"], df[col1])
    plt.plot(df["x"], df[col2])
    """
    col = get_fourier_col_name(1, "x", function_name="sin", seas_name="cat_period")
    assert df[col][0].round(1) == 0.0

    with pytest.raises(ValueError, match="periods and orders must have the same length."):
        fourier_series_multi_fcn(col_names=["tod", "tow"], periods=[24.0, 7.0], orders=[3, 3, 4], seas_names=["daily", "weekly"])


def test_signed_pow():
    assert signed_pow(-3, 2) == -9
    assert signed_pow(3, 2) == 9


def test_signed_pow_fcn():
    func = signed_pow_fcn(2)
    assert func(-2) == -4


def test_logistic():
    """Tests logistic function"""
    assert logistic(x=0.0) == 0.5
    # double the capacity
    assert logistic(x=0.0, capacity=2.0) == 1.0
    # increase the floor
    assert logistic(x=0.0, capacity=2.0, floor=1.0) == 2.0
    # growth rate doesn't value at the inflection point
    assert logistic(x=0.0, growth_rate=0.5, capacity=2.0, floor=1.0) == 2.0
    # translate the inflection point
    assert logistic(x=2.0, growth_rate=0.5, capacity=2.0, floor=1.0, inflection_point=2.0) == 2.0
    # lower value before the inflection point
    assert round(logistic(x=0.0, growth_rate=0.5, capacity=2.0, floor=1.0, inflection_point=2.0), 4) == 1.5379
    # symmetry around inflection point
    offset = 2.43
    inflection_point = 2.0
    lower = logistic(x=inflection_point - offset, inflection_point=inflection_point)
    mid = logistic(x=inflection_point, inflection_point=inflection_point)
    upper = logistic(x=inflection_point + offset, inflection_point=inflection_point)
    assert (upper - mid) == (mid - lower)


def test_get_logistic_func():
    """Tests get_logistic_func"""
    val = logistic(x=0.0, growth_rate=0.5, capacity=2.0, floor=1.0, inflection_point=2.0)
    func = get_logistic_func(growth_rate=0.5, capacity=2.0, floor=1.0, inflection_point=2.0)
    assert val == func(0.0)

    val = logistic(x=3.21, growth_rate=0.75, capacity=2.2, floor=1.1, inflection_point=-1.0)
    func = get_logistic_func(growth_rate=0.75, capacity=2.2, floor=1.1, inflection_point=-1.0)
    assert val == func(3.21)
