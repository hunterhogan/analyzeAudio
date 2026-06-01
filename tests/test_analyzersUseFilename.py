from __future__ import annotations

from analyzeAudio import analyzersUseFilename, audioAspects
import numpy
import pathlib
import pytest
import subprocess

@pytest.mark.parametrize(
	('filterChain', 'listDecibels', 'valueExpectedMean'), [('apsnr', [13.7, 17.3, 19.9], pytest.approx(16.966666666666665))]
)
def test_meanDB_parses_ffmpeg_stderr_lines(
	monkeypatch: pytest.MonkeyPatch, filterChain: str, listDecibels: list[float], valueExpectedMean: pytest.ApproxScalar
) -> None:
	class _MockSystemProcessResult:
		def __init__(self, stderr: bytes) -> None:
			self.stderr = stderr

	dictionaryCaptured: dict[str, object] = {}

	def _mockRun(commandLineFFmpeg: list[str], check: bool, stderr: int) -> _MockSystemProcessResult:
		dictionaryCaptured['commandLineFFmpeg'] = commandLineFFmpeg
		dictionaryCaptured['check'] = check
		dictionaryCaptured['stderr'] = stderr
		stringJoinedLines = '\n'.join([f'[Parsed_{filterChain}_97 @ 00000000] {valueDecibels} dB' for valueDecibels in listDecibels])
		return _MockSystemProcessResult(stderr=stringJoinedLines.encode('utf-8'))

	monkeypatch.setattr(analyzersUseFilename.subprocess, 'run', _mockRun)

	valueMean = analyzersUseFilename._meanDB('C:/audio/theta.wav', 'C:/audio/lambda.wav', filterChain)

	assert valueMean == valueExpectedMean, (
		f'_meanDB returned {valueMean}, expected {valueExpectedMean} for {filterChain=} and {listDecibels=}.'
	)
	assert dictionaryCaptured['check'] is True, f'subprocess.run received check={dictionaryCaptured["check"]}, expected True in _meanDB.'
	assert dictionaryCaptured['stderr'] == subprocess.PIPE, (
		f'subprocess.run received stderr={dictionaryCaptured["stderr"]}, expected subprocess.PIPE in _meanDB.'
	)
	assert '[0][1]apsnr' in dictionaryCaptured['commandLineFFmpeg'], (
		f"_meanDB command {dictionaryCaptured['commandLineFFmpeg']} did not include expected filter chain token '[0][1]apsnr'."
	)

@pytest.mark.parametrize(
	('pathFilename', 'dictionaryExpectedScalarValues', 'dictionaryExpectedArrayValues'),
	[
		(
			pathlib.PureWindowsPath(r'C:\audio\theta.wav'),
			{'DC_offset': 10.0, 'Zero_crossings': 17.0},
			{
				'mean': numpy.array([[2.5, 4.5], [6.5, 16.5]], dtype=numpy.float64),
				'variance': numpy.array([[3.5, 5.5], [7.5, 21.5]], dtype=numpy.float64),
				'I': numpy.array([-31.3, -28.7], dtype=numpy.float64),
				'LRA': numpy.array([4.7, 7.7], dtype=numpy.float64),
				'LRA.low': numpy.array([-42.1, -39.9], dtype=numpy.float64),
				'LRA.high': numpy.array([-21.3, -16.3], dtype=numpy.float64),
			},
		)
	],
)
def test_ffprobeShotgunAndCache_aggregates_features_and_reuses_cache(
	monkeypatch: pytest.MonkeyPatch,
	pathFilename: pathlib.PureWindowsPath,
	dictionaryExpectedScalarValues: dict[str, float],
	dictionaryExpectedArrayValues: dict[str, numpy.ndarray[tuple[int, int], numpy.dtype[numpy.float64]]],
) -> None:
	class _MockPopen:
		intCallCount = 0
		listCommands: list[list[str]] = []

		def __init__(
			self, commandLineFFprobe: list[str], stdin: int | None = None, stdout: int | None = None, stderr: int | None = None
		) -> None:
			type(self).intCallCount += 1
			type(self).listCommands.append(commandLineFFprobe)

		def communicate(self) -> tuple[bytes, bytes]:
			return b'{"frames": []}', b''

	def _mockPythonizeFFprobe(_stringJsonFFprobe: str) -> list[dict[str, dict[str, numpy.ndarray]]]:
		dictionaryStructured = {
			'aspectralstats': {
				'mean': numpy.array([[2.5, 4.5], [6.5, 16.5]], dtype=numpy.float64),
				'variance': numpy.array([[3.5, 5.5], [7.5, 21.5]], dtype=numpy.float64),
			},
			'r128': {
				'I': numpy.array([-31.3, -28.7], dtype=numpy.float64),
				'LRA': numpy.array([4.7, 7.7], dtype=numpy.float64),
				'LRA.low': numpy.array([-42.1, -39.9], dtype=numpy.float64),
				'LRA.high': numpy.array([-21.3, -16.3], dtype=numpy.float64),
			},
			'astats': {
				'lavfi.astats.Overall.DC_offset': numpy.array([[5.0, 7.0], [11.0, 13.0]], dtype=numpy.float64),
				'lavfi.astats.Overall.Zero_crossings': numpy.array([[13.0, 17.0], [19.0, 17.0]], dtype=numpy.float64),
			},
		}
		return [dictionaryStructured]

	analyzersUseFilename._ffprobeShotgunAndCache.cache_clear()
	monkeypatch.setattr(analyzersUseFilename.subprocess, 'Popen', _MockPopen)
	monkeypatch.setattr(analyzersUseFilename, 'pythonizeFFprobe', _mockPythonizeFFprobe)

	dictionaryFirstCall = analyzersUseFilename._ffprobeShotgunAndCache(pathFilename)
	dictionarySecondCall = analyzersUseFilename._ffprobeShotgunAndCache(pathFilename)

	assert _MockPopen.intCallCount == 1, (
		f'ffprobeShotgunAndCache called subprocess.Popen {_MockPopen.intCallCount} times, expected 1 call for cache reuse with {pathFilename=}.'
	)
	assert _MockPopen.listCommands, f'ffprobeShotgunAndCache did not capture any ffprobe command for {pathFilename=}.'
	assert any(('amovie=C' in token and ':/audio/theta.wav,' in token) for token in _MockPopen.listCommands[0]), (
		f'ffprobe command {_MockPopen.listCommands[0]} did not include expected Windows lavfi filename token for {pathFilename=}.'
	)
	for keyName, valueExpected in dictionaryExpectedScalarValues.items():
		assert keyName in dictionaryFirstCall, (
			f'ffprobeShotgunAndCache result keys {list(dictionaryFirstCall.keys())} did not include expected key {keyName!r}.'
		)
		assert dictionaryFirstCall[keyName] == pytest.approx(valueExpected), (
			f'ffprobeShotgunAndCache returned {dictionaryFirstCall[keyName]} for key {keyName!r}, expected {valueExpected} for {pathFilename=}.'
		)
	for keyName, arrayExpected in dictionaryExpectedArrayValues.items():
		assert keyName in dictionaryFirstCall, (
			f'ffprobeShotgunAndCache result keys {list(dictionaryFirstCall.keys())} did not include expected key {keyName!r}.'
		)
		assert numpy.array_equal(dictionaryFirstCall[keyName], arrayExpected), (
			f'ffprobeShotgunAndCache returned {dictionaryFirstCall[keyName]} for key {keyName!r}, expected {arrayExpected} for {pathFilename=}.'
		)
	assert dictionarySecondCall is dictionaryFirstCall, (
		f'Cached ffprobeShotgunAndCache call returned different object id for {pathFilename=}, expected same cached dictionary instance.'
	)

	analyzersUseFilename._ffprobeShotgunAndCache.cache_clear()

@pytest.mark.parametrize(
	('functionName', 'filterChainExpected', 'valueReturnedByMock'),
	[('getPSNRmean', 'apsnr', 23.7), ('getSDRmean', 'asdr', 29.3), ('getSI_SDRmean', 'asisdr', 31.7)],
)
def test_pairwise_db_wrappers_delegate_to_meanDB_with_expected_filter_chain(
	monkeypatch: pytest.MonkeyPatch, functionName: str, filterChainExpected: str, valueReturnedByMock: float
) -> None:
	dictionaryCaptured: dict[str, object] = {}

	def _mockMeanDB(pathFilenameAlfa: str, pathFilenameBeta: str, filterChain: str) -> float:
		dictionaryCaptured['pathFilenameAlfa'] = pathFilenameAlfa
		dictionaryCaptured['pathFilenameBeta'] = pathFilenameBeta
		dictionaryCaptured['filterChain'] = filterChain
		return valueReturnedByMock

	monkeypatch.setattr(analyzersUseFilename, '_meanDB', _mockMeanDB)

	functionAnalyzer = getattr(analyzersUseFilename, functionName)
	valueResult = functionAnalyzer('C:/audio/theta.wav', 'C:/audio/lambda.wav')

	assert valueResult == valueReturnedByMock, (
		f'{functionName} returned {valueResult}, expected {valueReturnedByMock} when _meanDB is mocked.'
	)
	assert dictionaryCaptured['pathFilenameAlfa'] == 'C:/audio/theta.wav', (
		f"{functionName} forwarded pathFilenameAlfa={dictionaryCaptured['pathFilenameAlfa']}, expected 'C:/audio/theta.wav'."
	)
	assert dictionaryCaptured['pathFilenameBeta'] == 'C:/audio/lambda.wav', (
		f"{functionName} forwarded pathFilenameBeta={dictionaryCaptured['pathFilenameBeta']}, expected 'C:/audio/lambda.wav'."
	)
	assert dictionaryCaptured['filterChain'] == filterChainExpected, (
		f'{functionName} delegated filterChain={dictionaryCaptured["filterChain"]}, expected {filterChainExpected}.'
	)

@pytest.mark.parametrize(
	('aspectName', 'functionName', 'dictionaryKeyName', 'valueExpected'),
	[
		('Zero crossings', 'analyzeZero_crossings', 'Zero_crossings', 2.3),
		('Zero-crossings rate', 'analyzeZero_crossings_rate', 'Zero_crossings_rate', 3.5),
		('DC offset', 'analyzeDCoffset', 'DC_offset', 5.7),
		('Dynamic range', 'analyzeDynamicRange', 'Dynamic_range', 7.9),
		('Signal entropy', 'analyzeSignalEntropy', 'Entropy', 11.3),
		('Duration-samples', 'analyzeNumber_of_samples', 'Number_of_samples', 13.7),
		('Peak dB', 'analyzePeak_level', 'Peak_level', 17.3),
		('RMS total', 'analyzeRMS_level', 'RMS_level', 19.7),
		('Crest factor', 'analyzeCrest_factor', 'Crest_factor', 23.3),
		('RMS peak', 'analyzeRMS_peak', 'RMS_peak', 29.3),
		('Abs_Peak_count', 'analyzeAbs_Peak_count', 'Abs_Peak_count', 97.3),
		('Bit_depth', 'analyzeBit_depth', 'Bit_depth', 101.7),
		('Flat_factor', 'analyzeFlat_factor', 'Flat_factor', 103.9),
		('Max_difference', 'analyzeMax_difference', 'Max_difference', 107.3),
		('Max_level', 'analyzeMax_level', 'Max_level', 109.7),
		('Mean_difference', 'analyzeMean_difference', 'Mean_difference', 113.3),
		('Min_difference', 'analyzeMin_difference', 'Min_difference', 127.9),
		('Min_level', 'analyzeMin_level', 'Min_level', 131.3),
		('Noise_floor', 'analyzeNoise_floor', 'Noise_floor', 137.3),
		('Noise_floor_count', 'analyzeNoise_floor_count', 'Noise_floor_count', 139.7),
		('Peak_count', 'analyzePeak_count', 'Peak_count', 149.3),
		('RMS_difference', 'analyzeRMS_difference', 'RMS_difference', 151.7),
		('RMS_trough', 'analyzeRMS_trough', 'RMS_trough', 157.3),
	],
)
def test_single_file_wrappers_return_dictionary_values_and_match_registration(
	monkeypatch: pytest.MonkeyPatch, aspectName: str, functionName: str, dictionaryKeyName: str, valueExpected: float
) -> None:
	dictionaryAspectsAnalyzed = {
		'Zero_crossings': 2.3,
		'Zero_crossings_rate': 3.5,
		'DC_offset': 5.7,
		'Dynamic_range': 7.9,
		'Entropy': 11.3,
		'Number_of_samples': 13.7,
		'Peak_level': 17.3,
		'RMS_level': 19.7,
		'Crest_factor': 23.3,
		'RMS_peak': 29.3,
		'Abs_Peak_count': 97.3,
		'Bit_depth': 101.7,
		'Flat_factor': 103.9,
		'Max_difference': 107.3,
		'Max_level': 109.7,
		'Mean_difference': 113.3,
		'Min_difference': 127.9,
		'Min_level': 131.3,
		'Noise_floor': 137.3,
		'Noise_floor_count': 139.7,
		'Peak_count': 149.3,
		'RMS_difference': 151.7,
		'RMS_trough': 157.3,
	}

	def _mockFFprobeShotgunAndCache(pathFilename: str) -> dict[str, float]:
		assert pathFilename == 'C:/audio/theta.wav', f"Mock ffprobeShotgunAndCache received {pathFilename}, expected 'C:/audio/theta.wav'."
		return dictionaryAspectsAnalyzed

	monkeypatch.setattr(analyzersUseFilename, '_ffprobeShotgunAndCache', _mockFFprobeShotgunAndCache)

	functionAnalyzer = getattr(analyzersUseFilename, functionName)
	valueResult = functionAnalyzer('C:/audio/theta.wav')

	assert valueResult == valueExpected, f'{functionName} returned {valueResult}, expected {valueExpected} for key {dictionaryKeyName!r}.'
	assert aspectName in audioAspects, f'audioAspects keys do not include {aspectName!r} for function {functionName}.'
	assert audioAspects[aspectName]['analyzer'] is functionAnalyzer, (
		f'audioAspects[{aspectName!r}] registered {audioAspects[aspectName]["analyzer"]}, expected {functionAnalyzer}.'
	)

@pytest.mark.parametrize(
	('functionName', 'dictionaryKeyName', 'valueExpectedMean'),
	[
		('analyzeSpectralCentroid', 'centroid', 43.7),
		('analyzeSpectralCrest', 'crest', 71.9),
		('analyzeSpectralDecrease', 'decrease', 83.9),
		('analyzeSpectralEntropy', 'entropy', 61.7),
		('analyzeSpectralFlatness', 'flatness', 67.3),
		('analyzeSpectralFlux', 'flux', 73.7),
		('analyzeSpectralKurtosis', 'kurtosis', 59.3),
		('analyzeSpectralMean', 'mean', 37.9),
		('analyzeSpectralRolloff', 'rolloff', 89.3),
		('analyzeSpectralSkewness', 'skewness', 53.9),
		('analyzeSpectralSlope', 'slope', 79.3),
		('analyzeSpectralSpread', 'spread', 47.3),
		('analyzeSpectralVariance', 'variance', 41.3),
	],
)
def test_aspectralstats_array_wrappers_return_dictionary_arrays(
	monkeypatch: pytest.MonkeyPatch, functionName: str, dictionaryKeyName: str, valueExpectedMean: float
) -> None:
	arrayExpected = numpy.array(
		[[valueExpectedMean - 6.0, valueExpectedMean - 2.0], [valueExpectedMean + 2.0, valueExpectedMean + 6.0]], dtype=numpy.float64
	)

	def _mockFFprobeShotgunAndCache(pathFilename: str) -> dict[str, numpy.ndarray[tuple[int, int], numpy.dtype[numpy.float64]]]:
		assert pathFilename == 'C:/audio/theta.wav', f"Mock ffprobeShotgunAndCache received {pathFilename}, expected 'C:/audio/theta.wav'."
		return {dictionaryKeyName: arrayExpected}

	monkeypatch.setattr(analyzersUseFilename, '_ffprobeShotgunAndCache', _mockFFprobeShotgunAndCache)

	functionAnalyzer = getattr(analyzersUseFilename, functionName)
	arrayResult = functionAnalyzer('C:/audio/theta.wav')

	assert numpy.array_equal(arrayResult, arrayExpected), (
		f'{functionName} returned {arrayResult.tolist()}, expected {arrayExpected.tolist()} for key {dictionaryKeyName!r}.'
	)

@pytest.mark.parametrize(
	('functionName', 'dictionaryKeyName', 'valueExpectedLast'),
	[
		('analyzeTruePeak', 'true_peak', 1.5),
		('analyzeLUFSMomentary', 'M', -21.0),
		('analyzeLUFSShortTerm', 'S', -23.0),
		('analyzeLUFSIntegrated', 'I', -28.7),
		('analyzeLRA', 'LRA', 31.1),
		('analyzeLUFSlow', 'LRA.low', -37.7),
		('analyzeLUFShigh', 'LRA.high', -19.3),
	],
)
def test_lufs_array_wrappers_return_dictionary_arrays(
	monkeypatch: pytest.MonkeyPatch, functionName: str, dictionaryKeyName: str, valueExpectedLast: float
) -> None:
	arrayExpected = numpy.array([valueExpectedLast - 2.0, valueExpectedLast], dtype=numpy.float64)

	def _mockFFprobeShotgunAndCache(pathFilename: str) -> dict[str, numpy.ndarray[tuple[int], numpy.dtype[numpy.float64]]]:
		assert pathFilename == 'C:/audio/theta.wav', f"Mock ffprobeShotgunAndCache received {pathFilename}, expected 'C:/audio/theta.wav'."
		return {dictionaryKeyName: arrayExpected}

	monkeypatch.setattr(analyzersUseFilename, '_ffprobeShotgunAndCache', _mockFFprobeShotgunAndCache)

	functionAnalyzer = getattr(analyzersUseFilename, functionName)
	arrayResult = functionAnalyzer('C:/audio/theta.wav')

	assert numpy.array_equal(arrayResult, arrayExpected), (
		f'{functionName} returned {arrayResult.tolist()}, expected {arrayExpected.tolist()} for key {dictionaryKeyName!r}.'
	)

@pytest.mark.parametrize(
	('aspectName', 'functionName', 'dictionaryKeyName', 'valueExpectedMean'),
	[
		('Spectral centroid mean', 'analyzeSpectralCentroidMean', 'centroid', 43.7),
		('Spectral crest mean', 'analyzeSpectralCrestMean', 'crest', 71.9),
		('Spectral decrease mean', 'analyzeSpectralDecreaseMean', 'decrease', 83.9),
		('Spectral entropy mean', 'analyzeSpectralEntropyMean', 'entropy', 61.7),
		('Spectral flatness mean', 'analyzeSpectralFlatnessMean', 'flatness', 67.3),
		('Spectral flux mean', 'analyzeSpectralFluxMean', 'flux', 73.7),
		('Spectral kurtosis mean', 'analyzeSpectralKurtosisMean', 'kurtosis', 59.3),
		('Power spectral density mean', 'analyzeSpectralMeanMean', 'mean', 37.9),
		('Spectral rolloff mean', 'analyzeSpectralRolloffMean', 'rolloff', 89.3),
		('Spectral skewness mean', 'analyzeSpectralSkewnessMean', 'skewness', 53.9),
		('Spectral slope mean', 'analyzeSpectralSlopeMean', 'slope', 79.3),
		('Spectral spread mean', 'analyzeSpectralSpreadMean', 'spread', 47.3),
		('Spectral variance mean', 'analyzeSpectralVarianceMean', 'variance', 41.3),
	],
)
def test_aspectralstats_mean_wrappers_return_array_means_and_match_registration(
	monkeypatch: pytest.MonkeyPatch, aspectName: str, functionName: str, dictionaryKeyName: str, valueExpectedMean: float
) -> None:
	arrayExpected = numpy.array(
		[[valueExpectedMean - 6.0, valueExpectedMean - 2.0], [valueExpectedMean + 2.0, valueExpectedMean + 6.0]], dtype=numpy.float64
	)

	def _mockFFprobeShotgunAndCache(pathFilename: str) -> dict[str, numpy.ndarray[tuple[int, int], numpy.dtype[numpy.float64]]]:
		assert pathFilename == 'C:/audio/theta.wav', f"Mock ffprobeShotgunAndCache received {pathFilename}, expected 'C:/audio/theta.wav'."
		return {dictionaryKeyName: arrayExpected}

	monkeypatch.setattr(analyzersUseFilename, '_ffprobeShotgunAndCache', _mockFFprobeShotgunAndCache)

	functionAnalyzer = getattr(analyzersUseFilename, functionName)
	valueResult = functionAnalyzer('C:/audio/theta.wav')

	assert valueResult == pytest.approx(valueExpectedMean), (
		f'{functionName} returned {valueResult}, expected {valueExpectedMean} for key {dictionaryKeyName!r}.'
	)
	assert aspectName in audioAspects, f'audioAspects keys do not include {aspectName!r} for function {functionName}.'
	assert audioAspects[aspectName]['analyzer'] is functionAnalyzer, (
		f'audioAspects[{aspectName!r}] registered {audioAspects[aspectName]["analyzer"]}, expected {functionAnalyzer}.'
	)

@pytest.mark.parametrize(
	('aspectName', 'functionName', 'dictionaryKeyName', 'valueExpectedLast'),
	[
		('true_peak maximum', 'analyzeTruePeakOverall', 'true_peak', 1.5),
		('LUFS momentary maximum', 'analyzeLUFSMomentaryOverall', 'M', -21.0),
		('LUFS short-term maximum', 'analyzeLUFSShortTermOverall', 'S', -23.0),
		('LUFS integrated', 'analyzeLUFSIntegratedOverall', 'I', -28.7),
		('LUFS loudness range', 'analyzeLRAOverall', 'LRA', 31.1),
		('LUFS low', 'analyzeLUFSlowOverall', 'LRA.low', -37.7),
		('LUFS high', 'analyzeLUFShighOverall', 'LRA.high', -19.3),
	],
)
def test_lufs_overall_wrappers_return_overall_value_and_match_registration(
	monkeypatch: pytest.MonkeyPatch, aspectName: str, functionName: str, dictionaryKeyName: str, valueExpectedLast: float
) -> None:
	arrayExpected = numpy.array([valueExpectedLast - 2.0, valueExpectedLast], dtype=numpy.float64)

	def _mockFFprobeShotgunAndCache(pathFilename: str) -> dict[str, numpy.ndarray[tuple[int], numpy.dtype[numpy.float64]]]:
		assert pathFilename == 'C:/audio/theta.wav', f"Mock ffprobeShotgunAndCache received {pathFilename}, expected 'C:/audio/theta.wav'."
		return {dictionaryKeyName: arrayExpected}

	monkeypatch.setattr(analyzersUseFilename, '_ffprobeShotgunAndCache', _mockFFprobeShotgunAndCache)

	functionAnalyzer = getattr(analyzersUseFilename, functionName)
	valueResult = functionAnalyzer('C:/audio/theta.wav')

	assert valueResult == pytest.approx(valueExpectedLast), (
		f'{functionName} returned {valueResult}, expected {valueExpectedLast} for key {dictionaryKeyName!r}.'
	)
	assert aspectName in audioAspects, f'audioAspects keys do not include {aspectName!r} for function {functionName}.'
	assert audioAspects[aspectName]['analyzer'] is functionAnalyzer, (
		f'audioAspects[{aspectName!r}] registered {audioAspects[aspectName]["analyzer"]}, expected {functionAnalyzer}.'
	)
