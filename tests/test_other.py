import pytest
from Z0Z_tools.pytestForYourUse import PytestFor_defineConcurrencyLimit, PytestFor_oopsieKwargsie

@pytest.mark.parametrize("nameOfTest,callablePytest", PytestFor_defineConcurrencyLimit())
def testConcurrencyLimit(nameOfTest, callablePytest):
	callablePytest()

@pytest.mark.parametrize("nameOfTest,callablePytest", PytestFor_oopsieKwargsie())
def testOopsieKwargsie(nameOfTest, callablePytest):
	callablePytest()
