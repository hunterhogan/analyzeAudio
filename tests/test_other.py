from __future__ import annotations

from collections.abc import Callable
from hunterMakesPy.tests.test_parseParameters import PytestFor_defineConcurrencyLimit, PytestFor_oopsieKwargsie
import pytest

@pytest.mark.parametrize("nameOfTest,callablePytest", PytestFor_defineConcurrencyLimit())
def testConcurrencyLimit(nameOfTest: str, callablePytest: Callable[[], None]) -> None:
	callablePytest()

@pytest.mark.parametrize("nameOfTest,callablePytest", PytestFor_oopsieKwargsie())
def testOopsieKwargsie(nameOfTest: str, callablePytest: Callable[[], None]) -> None:
	callablePytest()
