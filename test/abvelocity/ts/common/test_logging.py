"""
Test for logging.py
"""

from abvelocity.ts.common.constants import LOGGER_NAME
from abvelocity.ts.common.logging import LoggingLevelEnum, log_message
from testfixtures import LogCapture


def test_log_message():
    with LogCapture(LOGGER_NAME) as log_capture:
        log_message("Test log message.", LoggingLevelEnum.CRITICAL)
        log_capture.check((LOGGER_NAME, "CRITICAL", "Test log message."))
