"""Expected values are the accepted recordings in `tests.dataSamples.namesRegistered`."""
from __future__ import annotations

from analyzeAudio import getListAvailableAudioAspects, getListAvailableAudioContests, getListAvailableAudioMetadata
from tests import assertEqualTo
from tests.dataSamples.namesRegistered import listNamesAspect, listNamesContest, listNamesMetadata
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from hunterMakesPy import CallableFunction

@pytest.mark.parametrize('getListAvailable, expected', (
	pytest.param(getListAvailableAudioAspects, listNamesAspect, id='getListAvailableAudioAspects'),
	pytest.param(getListAvailableAudioContests, listNamesContest, id='getListAvailableAudioContests'),
	pytest.param(getListAvailableAudioMetadata, listNamesMetadata, id='getListAvailableAudioMetadata'),
))
def test_getListAvailable(getListAvailable: CallableFunction[[], list[str]], expected: list[str]) -> None:
	assertEqualTo(getListAvailable(), expected, getListAvailable.__name__)
