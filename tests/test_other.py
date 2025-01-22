import pytest
from Z0Z_tools.pytest_parseParameters import makeTestSuiteConcurrencyLimit, makeTestSuiteOopsieKwargsie
from Z0Z_tools import defineConcurrencyLimit, oopsieKwargsie

def test_oopsieKwargsie():
    dictionaryTests = makeTestSuiteOopsieKwargsie(oopsieKwargsie)
    for testName, testFunction in dictionaryTests.items():
        testFunction()

def test_defineConcurrencyLimit():
    dictionaryTests = makeTestSuiteConcurrencyLimit(defineConcurrencyLimit)
    for testName, testFunction in dictionaryTests.items():
        testFunction()
