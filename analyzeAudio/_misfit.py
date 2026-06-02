from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Iterable
	from os import PathLike
	from pathlib import PurePath

def dataTabularTOpathFilenameDelimited(pathFilename: PathLike[Any] | PurePath, tableRows: Iterable[Iterable[Any]], tableColumns: Iterable[Any], delimiterOutput: str = '\t') -> None:
	r"""Write tabular rows to a delimited text file.

	You can use this function to write `tableRows` and `tableColumns` to
	`pathFilename` with `delimiterOutput`. `analyzeAudio.analyzeAudioListPathFilenames` [1]
	returns row data that this function can write directly.

	Parameters
	----------
	pathFilename : PathLike[Any] | PurePath
		Path of the output text file.
	tableRows : Iterable[Iterable[Any]]
		Row sequence to write after the header row.
	tableColumns : Iterable[Any]
		Column label sequence for the optional header row.
	delimiterOutput : str = '\t'
		Text delimiter inserted between adjacent cells.

	Examples
	--------
	Write rows returned by `analyzeAudio.analyzeAudioListPathFilenames` [1].

	```python
	from analyzeAudio import dataTabularTOpathFilenameDelimited
	from analyzeAudio.analyze import analyzeAudioListPathFilenames
	import pathlib

	lPFn = list(pathlib.Path('/apps/analyzeAudio/tests/dataSamples').rglob('test*.wav'))
	singleTargetFloats = ['Crest factor', 'Spectral flatness']
	rows = analyzeAudioListPathFilenames(lPFn, singleTargetFloats)

	dataTabularTOpathFilenameDelimited(
		lPFn[0].parent.parent.parent / 'l073.tab',
		rows,
		['pathFilename', *singleTargetFloats],
	)
	```

	References
	----------
	[1] `analyzeAudio.analyzeAudioListPathFilenames`

	"""
	with open(pathFilename, 'w', newline='', encoding='utf-8') as writeStream:  # noqa: PTH123
		# Write headers if they exist
		if tableColumns:
			writeStream.write(delimiterOutput.join(map(str, tableColumns)) + '\n')

		# Write rows
		writeStream.writelines(delimiterOutput.join(map(str, row)) + '\n' for row in tableRows)
