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
import pytest
from abvelocity.ts.get_data.ts_metrics_config import FREQ_TO_SQL_UNIT, TSMetricsConfig, build_time_expr, parse_col_expr

# ---------------------------------------------------------------------------
# parse_col_expr
# ---------------------------------------------------------------------------


def test_parse_col_expr_trino():
    assert parse_col_expr("event_ts", "%Y-%m-%d", "trino") == ("DATE_PARSE(event_ts, '%Y-%m-%d')")


def test_parse_col_expr_mysql():
    assert parse_col_expr("event_ts", "%Y-%m-%d", "mysql") == ("STR_TO_DATE(event_ts, '%Y-%m-%d')")


def test_parse_col_expr_duckdb():
    assert parse_col_expr("event_ts", "%Y-%m-%d", "duckdb") == ("STRPTIME(event_ts, '%Y-%m-%d')")


def test_parse_col_expr_spark_converts_format():
    result = parse_col_expr("event_ts", "%Y-%m-%d %H:%M:%S", "spark")
    assert result == "TO_TIMESTAMP(event_ts, 'yyyy-MM-dd HH:mm:ss')"


def test_parse_col_expr_spark_custom_separator():
    result = parse_col_expr("event_ts", "%Y-%m-%d:%H", "spark")
    assert result == "TO_TIMESTAMP(event_ts, 'yyyy-MM-dd:HH')"


# ---------------------------------------------------------------------------
# build_time_expr — freq aliases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "freq,expected_unit",
    [
        # Second
        ("s", "second"),
        ("S", "second"),
        # Minute — legacy and pandas ≥ 2.2
        ("T", "minute"),
        ("min", "minute"),
        # Hour — legacy and pandas ≥ 2.2
        ("H", "hour"),
        ("h", "hour"),
        # Day
        ("D", "day"),
        ("d", "day"),
        # Week
        ("W", "week"),
        # Month variants
        ("M", "month"),
        ("MS", "month"),
        ("ME", "month"),
        # Quarter variants
        ("Q", "quarter"),
        ("QS", "quarter"),
        ("QE", "quarter"),
        # Year variants
        ("Y", "year"),
        ("A", "year"),
        ("YS", "year"),
        ("YE", "year"),
    ],
)
def test_build_time_expr_freq_aliases(freq, expected_unit):
    expr = build_time_expr("ts", freq, "trino")
    assert expr == f"DATE_TRUNC('{expected_unit}', ts)"


def test_build_time_expr_strips_offset_qualifier():
    assert build_time_expr("ts", "QS-JAN", "trino") == "DATE_TRUNC('quarter', ts)"
    assert build_time_expr("ts", "QS-APR", "trino") == "DATE_TRUNC('quarter', ts)"
    assert build_time_expr("ts", "YS-JUL", "trino") == "DATE_TRUNC('year', ts)"


def test_build_time_expr_unsupported_freq_raises():
    with pytest.raises(ValueError, match="Unsupported freq"):
        build_time_expr("ts", "X", "trino")


def test_build_time_expr_unsupported_dialect_raises():
    with pytest.raises(ValueError, match="Unsupported dialect"):
        build_time_expr("ts", "D", "postgres")


# ---------------------------------------------------------------------------
# build_time_expr — time_format (string columns)
# ---------------------------------------------------------------------------


def test_build_time_expr_with_time_format_trino():
    expr = build_time_expr("event_ts", "D", "trino", time_format="%Y-%m-%d:%H")
    assert expr == "DATE_TRUNC('day', DATE_PARSE(event_ts, '%Y-%m-%d:%H'))"


def test_build_time_expr_with_time_format_mysql():
    expr = build_time_expr("event_ts", "D", "mysql", time_format="%Y-%m-%d")
    assert expr == "DATE_TRUNC('day', STR_TO_DATE(event_ts, '%Y-%m-%d'))"


def test_build_time_expr_with_time_format_duckdb():
    expr = build_time_expr("event_ts", "h", "duckdb", time_format="%Y-%m-%d %H")
    assert expr == "DATE_TRUNC('hour', STRPTIME(event_ts, '%Y-%m-%d %H'))"


def test_build_time_expr_with_time_format_spark():
    expr = build_time_expr("event_ts", "D", "spark", time_format="%Y-%m-%d")
    assert expr == "DATE_TRUNC('day', TO_TIMESTAMP(event_ts, 'yyyy-MM-dd'))"


def test_build_time_expr_without_time_format():
    expr = build_time_expr("event_ts", "D", "trino")
    assert expr == "DATE_TRUNC('day', event_ts)"


# ---------------------------------------------------------------------------
# TSMetricsConfig
# ---------------------------------------------------------------------------


def test_ts_metrics_config_auto_builds_time_expr():
    cfg = TSMetricsConfig(time_col="event_ts", freq="D")
    assert cfg.time_expr == "DATE_TRUNC('day', event_ts)"
    assert cfg.dialect == "trino"


def test_ts_metrics_config_with_time_format():
    cfg = TSMetricsConfig(time_col="event_ts", freq="h", time_format="%Y-%m-%d:%H")
    assert cfg.time_expr == "DATE_TRUNC('hour', DATE_PARSE(event_ts, '%Y-%m-%d:%H'))"


def test_ts_metrics_config_custom_time_expr_not_overwritten():
    custom = "TO_DATE(CAST(date_int AS VARCHAR), 'YYYYMMDD')"
    cfg = TSMetricsConfig(time_col="date_int", freq="D", time_expr=custom)
    assert cfg.time_expr == custom


def test_ts_metrics_config_spark_dialect():
    cfg = TSMetricsConfig(time_col="ts", freq="ME", dialect="spark", time_format="%Y-%m-%d")
    assert cfg.time_expr == "DATE_TRUNC('month', TO_TIMESTAMP(ts, 'yyyy-MM-dd'))"


def test_ts_metrics_config_time_alias_default():
    cfg = TSMetricsConfig(time_col="ts", freq="D")
    assert cfg.time_alias == "ts"


def test_ts_metrics_config_all_freq_aliases_in_map():
    for freq in FREQ_TO_SQL_UNIT:
        cfg = TSMetricsConfig(time_col="ts", freq=freq)
        assert cfg.time_expr is not None
