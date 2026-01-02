import logging
from rtcog.utils.log import get_logger, set_logger


def test_get_logger():
    logger = get_logger()
    assert isinstance(logger, logging.Logger)
    assert logger.name == "GENERAL"
    assert len(logger.handlers) >= 2


def test_get_logger_idempotent():
    logger1 = get_logger()
    logger2 = get_logger()
    assert logger1 is logger2


def test_set_logger_debug():
    logger = set_logger(debug=True)
    assert logger.level == logging.DEBUG


def test_set_logger_silent():
    logger = set_logger(silent=True)
    assert logger.level == logging.CRITICAL


def test_set_logger_default():
    logger = set_logger()
    assert logger.level == logging.INFO


def test_set_logger_debug_overrides_silent():
    logger = set_logger(debug=True, silent=True)
    assert logger.level == logging.DEBUG