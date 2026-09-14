import logging

from jorgpt.logging_utils import get_logger


def test_get_logger_returns_logger():
    logger = get_logger("jorgpt.test")

    assert isinstance(logger, logging.Logger)


def test_get_logger_does_not_duplicate_handlers():
    logger = get_logger("jorgpt.test.duplicate")

    initial_handlers = len(logger.handlers)

    logger = get_logger("jorgpt.test.duplicate")

    assert len(logger.handlers) == initial_handlers
