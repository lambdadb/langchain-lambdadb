"""Pytest configuration for integration tests."""

from typing import Any, Generator

import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo
) -> Generator[Any, None, Any]:
    """Store test report on the item so fixtures can see pass/fail."""
    outcome: Any = yield
    rep = outcome.get_result()
    setattr(item, "rep_" + rep.when, rep)
