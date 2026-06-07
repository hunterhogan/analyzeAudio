"""Analyzers that use the filename of an audio file to analyze its audio data."""
from __future__ import annotations

from analyzeAudio.analyzersUseFilename._pythonator import pythonizeFFprobe
from functools import cache
from operator import getitem
from typing import Any, TYPE_CHECKING
import pathlib
import subprocess  # noqa: S404

if TYPE_CHECKING:
	from analyzeAudio import ArrayChannelData, ArrayOverallData
	from os import PathLike

@cache
def ffprobeAllInclusiveCache(pathFilename: str | PathLike[Any]) -> dict[str, ArrayChannelData | ArrayOverallData]:
	"""I use this shared extractor to collect scalar audio aspects from one analysis pass.

	(AI generated docstring)

	I use this function to convert one structured analysis result into a dictionary of scalar and
	array audio aspects.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	dictionaryAspects : dict[str, ArrayChannelData | ArrayOverallData]
		Dictionary mapping aspect identifiers to array numeric values.
	"""
	# TODO Investigate, why `PureWindowsPath`?
	# `as_posix` because using lavfi bypasses the CLI sanitation/standardization functions, AND lavfi
	# either never works with NT paths or doesn't always work with NT paths, but POSIX is always safe
	# IF escaped properly. Does this work in POSIX filesystems? IDK. The "contest" aspects, like
	# SI-SDR use a different FFmpeg call that treats the filenames with
	# `str(pathlib.Path(pathFilenameBeta))`.
	pFn = pathlib.PureWindowsPath(pathFilename)
	# for lavfi amovie/movie, the colons after driveLetter letters need to be escaped twice.
	lavfiPathFilename = pFn.drive.replace(":", "\\\\:") + pathlib.PureWindowsPath(pFn.root, pFn.relative_to(pFn.anchor)).as_posix()

	filterChain: list[str] = []
	filterChain += ["aspectralstats"]
	# by default length=0.05, 50ms. Set to 0.1, 100ms to match ebur128.
	# TODO FFmpeg might have a bug. per-channel `Abs_Peak_count` is not inserted in the metadata, but it is in the parsed_stats summary.
	filterChain += ["astats=metadata=1:length=0.1:measure_perchannel=all:measure_overall=Number_of_samples+RMS_level+Abs_Peak_count"]
	filterChain += ["ebur128=metadata=1:dualmono=true:framelog=verbose:peak=true"]

	entriesFFprobe: list[str] = ["frame_tags"]

	commandLineFFprobe: list[str] = [
		"ffprobe"
		, "-hide_banner"
		, "-f"
		, "lavfi"
		, f"amovie={lavfiPathFilename},{','.join(filterChain)}"
		, "-show_entries"
		, ':'.join(entriesFFprobe)
		, "-output_format"
		, "json=compact=1"
	]

	systemProcessFFprobe = subprocess.Popen(commandLineFFprobe, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	stdoutFFprobe, _DISCARDstderr = systemProcessFFprobe.communicate()
	FFprobeStructured = getitem(pythonizeFFprobe(stdoutFFprobe.decode('utf-8')), -1)

	dictionaryAspects: dict[str, ArrayChannelData | ArrayOverallData] = {}
	if 'aspectralstats' in FFprobeStructured:
		"""No matter how many channels, each keyName is `numpy.ndarray[tuple[int, int], numpy.dtype[numpy.float64]]`
		where `tuple[int, int]` is (channel, frame)
		"""
		dictionaryAspects.update(FFprobeStructured['aspectralstats'])
	if 'r128' in FFprobeStructured:
		dictionaryAspects.update(FFprobeStructured['r128'])
		# index -1 is the cumulative value for LUFS I, low, and high and peak; plus, the array has 3 significant digits instead of the summary's 1.
	if 'astats' in FFprobeStructured:
		dictionaryAspects.update(FFprobeStructured['astats'])
		# TODO Crest_factor "standard ratio of peak to RMS level (note: not in dB)"

		# TODO Bit_depth: 'Bit_depth', 'Bit_depth2', 'Bit_depth3', 'Bit_depth4',

		# If I ran two passes of the filter, I could force per-frame values for all aspects with `reset=1`. astats is pretty fast because there are no transformations.

	return dictionaryAspects
