"""Expected values come from the docstring of `dataTabularTOpathFilenameDelimited`: an optional header row of `tableColumns`, then one line for each row of `tableRows`, each cell converted with `str` and separated by `delimiterOutput`."""
from __future__ import annotations

from analyzeAudio import dataTabularTOpathFilenameDelimited
from tests import assertEqualTo
from typing import Any, TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from pathlib import Path

@pytest.mark.parametrize('tableRows, tableColumns, delimiterOutput, expected', (
	pytest.param((tableRowsAspects := [['Lagos.wav', 13, 0.5], ['Lima.wav', 21, None]]), (tableColumnsAspects := ['pathFilename', 'Tempo mean', 'NISQA mean'])
		, '\t', 'pathFilename\tTempo mean\tNISQA mean\nLagos.wav\t13\t0.5\nLima.wav\t21\tNone\n', id='tableColumns-tab'),
	pytest.param(tableRowsAspects, tableColumnsAspects, ',', 'pathFilename,Tempo mean,NISQA mean\nLagos.wav,13,0.5\nLima.wav,21,None\n', id='tableColumns-comma'),
	pytest.param(tableRowsAspects, [], '\t', 'Lagos.wav\t13\t0.5\nLima.wav\t21\tNone\n', id='tableColumnsEmpty-tab'),
	pytest.param([], tableColumnsAspects, '\t', 'pathFilename\tTempo mean\tNISQA mean\n', id='tableRowsEmpty-tab'),
))
def test_dataTabularTOpathFilenameDelimited(tmp_path: Path, tableRows: list[list[Any]], tableColumns: list[str], delimiterOutput: str, expected: str) -> None:
	pathFilename: Path = tmp_path / 'aspects.tab'
	assert dataTabularTOpathFilenameDelimited(pathFilename, tableRows, tableColumns, delimiterOutput) is pathFilename, (
		f'{dataTabularTOpathFilenameDelimited.__name__}({pathFilename!r}, ...) did not return {pathFilename!r}.')
	assertEqualTo(pathFilename.read_text(encoding='utf-8'), expected, dataTabularTOpathFilenameDelimited.__name__, pathFilename.name, tableRows, tableColumns, delimiterOutput)
