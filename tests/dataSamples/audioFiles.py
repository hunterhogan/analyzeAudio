"""Audio files in `tests/dataSamples`, by sample label, as paths relative to that directory.

A label of one file is the file's stem, with each decimal number spelled as a count of a named unit: the
stem `ch2_44100_04s_LUFS69.5_AbsGate` has the label `ch2_44100_04s_LUFS139Halves_AbsGate`, and the stem
`ch2_44100_7.1s_s16le` has the label `ch2_44100_71TenthsSeconds_s16le`. A label of a contest names the stem
that both files separate and the quality of the comparand's separation.

Every file in `SpeakSoftly_BrokenMan60sec` has the sample rate `sampleRateSpeakSoftly`. Consumers do not
mutate these values.
"""
from __future__ import annotations

lookupPathFilenamesAudio: dict[str, str] = {
	'ch1_16000_09s_s32le_Clipping': 'ch1_16000_09s_s32le_Clipping.wav',
	'ch2_44100_04s_LUFS10_RelGate': 'ch2_44100_04s_LUFS10_RelGate.wav',
	'ch2_44100_04s_LUFS139Halves_AbsGate': 'ch2_44100_04s_LUFS69.5_AbsGate.wav',
	'ch2_44100_05s_s16le': 'ch2_44100_05s_s16le.wav',
	'ch2_44100_09s_LUFS20_birdsPink': 'ch2_44100_09s_LUFS20_birdsPink.wav',
	'ch2_44100_71TenthsSeconds_s16le': 'ch2_44100_7.1s_s16le.wav',
	'ch2_48000_04s_LUFS10_RelGate': 'ch2_48000_04s_LUFS10_RelGate.wav',
	'ch2_48000_04s_LUFS139Halves_AbsGate': 'ch2_48000_04s_LUFS69.5_AbsGate.wav',
	'ch2_48000_63TenthsSeconds_s16le': 'ch2_48000_6.3s_s16le.wav',
}

lookupPathFilenamesContest: dict[str, tuple[str, str]] = {
	'bassBad': ('SpeakSoftly_BrokenMan60sec/reference_bass_Hz44100.flac', 'SpeakSoftly_BrokenMan60sec/comparand_bass_bad_Hz44100.flac'),
	'bassGood': ('SpeakSoftly_BrokenMan60sec/reference_bass_Hz44100.flac', 'SpeakSoftly_BrokenMan60sec/comparand_bass_good_Hz44100.flac'),
	'drumsBad': ('SpeakSoftly_BrokenMan60sec/reference_drums_Hz44100.flac', 'SpeakSoftly_BrokenMan60sec/comparand_drums_bad_Hz44100.flac'),
	'drumsGood': ('SpeakSoftly_BrokenMan60sec/reference_drums_Hz44100.flac', 'SpeakSoftly_BrokenMan60sec/comparand_drums_good_Hz44100.flac'),
	'otherBad': ('SpeakSoftly_BrokenMan60sec/reference_other_Hz44100.flac', 'SpeakSoftly_BrokenMan60sec/comparand_other_bad_Hz44100.flac'),
	'otherGood': ('SpeakSoftly_BrokenMan60sec/reference_other_Hz44100.flac', 'SpeakSoftly_BrokenMan60sec/comparand_other_good_Hz44100.flac'),
	'vocalsBad': ('SpeakSoftly_BrokenMan60sec/reference_vocals_Hz44100.flac', 'SpeakSoftly_BrokenMan60sec/comparand_vocals_bad_Hz44100.flac'),
	'vocalsGood': ('SpeakSoftly_BrokenMan60sec/reference_vocals_Hz44100.flac', 'SpeakSoftly_BrokenMan60sec/comparand_vocals_good_Hz44100.flac'),
}
"""The reference stem, then the comparand that estimates it."""

pathFilenameMixture: str = 'SpeakSoftly_BrokenMan60sec/reference_mixture_Hz44100.flac'
"""The mixture from which the comparands in `lookupPathFilenamesContest` were separated."""

sampleRateSpeakSoftly: int = 44100
