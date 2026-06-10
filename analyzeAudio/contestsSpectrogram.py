# ruff: noqa: D100 D103
from __future__ import annotations

from analyzeAudio import BleedFull, BleedFullArray, ParametersMelSpectrogram
from analyzeAudio.registry import registrationAudioContest
from typing import TYPE_CHECKING
from typing_extensions import Unpack
import librosa
import numpy
import sys
import warnings

if TYPE_CHECKING:
	from analyzeAudio import SpectrogramMagnitude

# TODO Figure out how to cache intermediate objects in this module.

def analyzeBleedFullMelDB(
		spectrogramMagnitudeAlfa: SpectrogramMagnitude
		, spectrogramMagnitudeBeta: SpectrogramMagnitude
		, **keywordArguments: Unpack[ParametersMelSpectrogram]
	) -> BleedFullArray:
	"""Separate mel-scaled dB excess and deficit between two spectrograms.

	You can use this function to compare two magnitude spectrograms as source-separation balance data.
	The function returns one array for places where `spectrogramMagnitudeBeta` has more mel-scaled dB
	energy than `spectrogramMagnitudeAlfa` and one array for places where `spectrogramMagnitudeBeta`
	has less mel-scaled dB energy than `spectrogramMagnitudeAlfa` [1][2].

	Parameters
	----------
	spectrogramMagnitudeAlfa : SpectrogramMagnitude
		Baseline magnitude spectrogram for the comparison.
	spectrogramMagnitudeBeta : SpectrogramMagnitude
		Second magnitude spectrogram whose added and missing mel-scaled dB content is measured against
		`spectrogramMagnitudeAlfa`.

	Returns
	-------
	bleedFullArray : BleedFullArray
		Named tuple containing `arrayBleed` with positive dB differences and `arrayFull` with negative
		dB differences. Zero dB differences are omitted.

	Mathematics
	-----------
	mel dB difference split : equation
	```
		Let Xₐ ≜ `spectrogramMagnitudeAlfa`,  Xᵦ ≜ `spectrogramMagnitudeBeta`
			F ≜ mel-scaled linear transformation matrix

		Mₐ = F · Xₐ
		Mᵦ = F · Xᵦ

		Aᵢ = 20 × log₁₀(Mₐᵢ)   ∀ Mₐᵢ ≠ 0, ∀ Aᵢ < |80.0|
		Bᵢ = 20 × log₁₀(Mᵦᵢ)   ∀ Mᵦᵢ ≠ 0, ∀ Bᵢ < |80.0|

		Δ = B − A

		Β = {Δᵢ ∈ Δ | Δᵢ > 0}
		Φ = {Δᵢ ∈ Δ | Δᵢ < 0}

		where  Β ≜ `arrayBleed`,  Φ ≜ `arrayFull`
	```

	Other Parameters
	----------------
	dtype : DTypeLike = numpy.float32
		Data type for the mel spectrograms and the returned difference arrays.
	fmax : float | None = None
		Maximum frequency (in hertz) included in the mel spectrograms. If `None`, the spectrograms are
		computed up to the Nyquist frequency.
	fmin : float = 0
		Minimum frequency (in hertz) included in the mel spectrograms.
	hop_length : int = 1024
		Number of samples between successive frames.
	htk : bool = False
		Whether to use HTK formula for mel scale.
	n_fft : int = 4096
		Length of the FFT window.
	n_mels : int = 512
		Number of Mel bands to generate.
	norm : float | Literal['slaney'] | None = "slaney"
		Normalization method for the mel spectrogram.
	power : float = 1.0
		Exponent for the magnitude spectrogram.
	sr : int = 44100
		Sample rate in hertz used to build the mel filter bank.
	top_db : float | None = 80.0
		Maximum decibel range retained after amplitude-to-decibel conversion.
	win_length : int = None
		Windowing function length for the FFT.
	window : str | tuple[Any, ...] | float | Callable[[int], ndarray] | ArrayLike = "hann"
		Windowing function for the FFT.

	References
	----------
	[1] csteinmetz1/auraloss issue #79. Enhancement ? New metric for source
		separation, measuring separately bleed and fullness in separated audio.
		https://github.com/csteinmetz1/auraloss/issues/79
	[2] ZFTurbo and jarredou. Music-Source-Separation-Training `bleed_full` metric.
		https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/c0197a0b2f1fffa8631779e1e92835a2e24d1c99/utils/metrics.py#L304-L385
	[3] librosa.feature.melspectrogram.
		https://librosa.org/doc/latest/generated/librosa.feature.melspectrogram.html
	[4] librosa.amplitude_to_db.
		https://librosa.org/doc/latest/generated/librosa.amplitude_to_db.html
	"""
	parametersMelSpectrogram = ParametersMelSpectrogram(
		dtype=keywordArguments.get('dtype', numpy.float32)
		, fmax=keywordArguments.get('fmax')
		, fmin=keywordArguments.get('fmin', 0)
		, hop_length=keywordArguments.get('hop_length', 1024)
		, htk=keywordArguments.get('htk', False)
		, n_fft=keywordArguments.get('n_fft', 4096)
		, n_mels=keywordArguments.get('n_mels', 512)
		, norm=keywordArguments.get('norm', "slaney")
		, power=keywordArguments.get('power', 1.0)
		, sr=keywordArguments.get('sr', 44100)
		, win_length=keywordArguments.get('win_length', keywordArguments.get('n_fft', 4096))
		, window=keywordArguments.get('window', "hann")
	)

	top_db: float | None = keywordArguments.get('top_db', 80.0)

	with warnings.catch_warnings(record=True) as warningMessages:
		warnings.simplefilter('always', UserWarning)
		spectrogramMagnitudeAlfa = librosa.feature.melspectrogram(S=spectrogramMagnitudeAlfa, **parametersMelSpectrogram)
		spectrogramMagnitudeBeta = librosa.feature.melspectrogram(S=spectrogramMagnitudeBeta, **parametersMelSpectrogram)
	for warningMessage in warningMessages:
		if str(warningMessage.message).startswith('Empty filters detected in mel frequency basis.'):
			message: str = (
				f'While computing `analyzeBleedFullMelDB({spectrogramMagnitudeAlfa.shape = }, {spectrogramMagnitudeBeta.shape = })` because the difference between the maximum frequency and minimum frequency is too small for the number of mel bands, some mel bands were empty.'
			)
			sys.stderr.write(message + '\n')
		else:
			warnings.warn(str(warningMessage.message), warningMessage.category, stacklevel=2)
	spectrogramMagnitudeAlfa = librosa.amplitude_to_db(spectrogramMagnitudeAlfa, ref=1.0, top_db=top_db)  # pyright: ignore[reportUnknownMemberType]
	spectrogramMagnitudeBeta = librosa.amplitude_to_db(spectrogramMagnitudeBeta, ref=1.0, top_db=top_db)  # pyright: ignore[reportUnknownMemberType]

	return _bleedFullArrays(spectrogramMagnitudeAlfa, spectrogramMagnitudeBeta)

def _bleedFullArrays(spectrogramAlfa: SpectrogramMagnitude, spectrogramBeta: SpectrogramMagnitude) -> BleedFullArray:
	arrayDifferences = spectrogramBeta - spectrogramAlfa

	return BleedFullArray(arrayBleed=arrayDifferences[0 < arrayDifferences], arrayFull=arrayDifferences[arrayDifferences < 0])

def analyzeBleedFullMelDBMean(
		spectrogramMagnitudeAlfa: SpectrogramMagnitude
		, spectrogramMagnitudeBeta: SpectrogramMagnitude
		, **keywordArguments: Unpack[ParametersMelSpectrogram]
) -> BleedFull:
	"""Score mean mel-scaled dB-magnitude excess and deficit between two spectrograms.

	You can use this function to summarize source-separation balance as two higher-is-better scores.
	The function reports `bleed` from the mean positive dB-magnitude difference and `full` from the
	mean negative dB-magnitude difference, so added content and missing content remain separate
	[1][2].

	Parameters
	----------
	spectrogramMagnitudeAlfa : SpectrogramMagnitude
		Baseline magnitude spectrogram for the comparison.
	spectrogramMagnitudeBeta : SpectrogramMagnitude
		Second magnitude spectrogram whose added and missing mel-scaled dB-magnitude content is scored
		against `spectrogramMagnitudeAlfa`.

	Returns
	-------
	bleedFull : BleedFull
		Named tuple containing `bleed` and `full` reciprocal scores. Each score is `100 / (1 +
		|meanDifference|)` when the matching difference class exists and `0.0` when the matching
		difference class is empty.

	Mathematics
	-----------
	score mapping : equation
	```
		Let Δ⁺ ≜ positive mel-scaled dB-magnitude differences
			Δ⁻ ≜ negative mel-scaled dB-magnitude differences

		bleed = 100 / (1 + |mean(Δ⁺)|)
		full = 100 / (1 + |mean(Δ⁻)|)
	```

	Score Interpretation
	--------------------
	The reciprocal scale maps a zero mean magnitude to `100.0`. Because zero dB-magnitude differences
	are excluded from the selected difference class, nonempty difference classes usually score below
	`100.0`. Larger average dB-magnitude differences reduce the score toward `0.0`.

	See Also
	--------
	analyzeBleedFullMelDB
		Return the underlying positive and negative mel-scaled dB-magnitude difference arrays.

	References
	----------
	[1] csteinmetz1/auraloss issue #79. Enhancement ? New metric for source
		separation, measuring separately bleed and fullness in separated audio.
		https://github.com/csteinmetz1/auraloss/issues/79
	[2] ZFTurbo. Music-Source-Separation-Training `bleed_full` metric.
		https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/c0197a0b2f1fffa8631779e1e92835a2e24d1c99/utils/metrics.py#L304-L385
	[3] numpy.mean.
		https://numpy.org/doc/stable/reference/generated/numpy.mean.html
	"""
	bf: BleedFullArray = analyzeBleedFullMelDB(spectrogramMagnitudeAlfa, spectrogramMagnitudeBeta, **keywordArguments)

	if 0 < bf.arrayBleed.size:
		bleed: float = numpy.mean(bf.arrayBleed).item()
		bleed = 100 / (1 + abs(bleed))
	else:
		bleed = 0.0

	if 0 < bf.arrayFull.size:
		full: float = numpy.mean(bf.arrayFull).item()
		full = 100 / (1 + abs(full))
	else:
		full = 0.0

	return BleedFull(bleed=bleed, full=full)

@registrationAudioContest('Bleedless Mel-scaled dB mean')
def analyzeBleedlessMelDBMean(
		spectrogramMagnitudeAlfa: SpectrogramMagnitude
		, spectrogramMagnitudeBeta: SpectrogramMagnitude
		, **keywordArguments: Unpack[ParametersMelSpectrogram]
) -> float:
	return analyzeBleedFullMelDBMean(spectrogramMagnitudeAlfa, spectrogramMagnitudeBeta, **keywordArguments).bleed

@registrationAudioContest('Fullness Mel-scaled dB mean')
def analyzeFullnessMelDBMean(
		spectrogramMagnitudeAlfa: SpectrogramMagnitude
		, spectrogramMagnitudeBeta: SpectrogramMagnitude
		, **keywordArguments: Unpack[ParametersMelSpectrogram]
) -> float:
	return analyzeBleedFullMelDBMean(spectrogramMagnitudeAlfa, spectrogramMagnitudeBeta, **keywordArguments).full
