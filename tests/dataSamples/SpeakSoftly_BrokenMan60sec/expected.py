from __future__ import annotations

expectedFilename: dict[str, dict[tuple[str, str], float]] = {
	'analyzePSNRmean': {
		('reference_bass.wav', 'comparand_bass_bad.wav'): 171.2265
		, ('reference_bass.wav', 'comparand_bass_good.wav'): 167.9665
		, ('reference_drums.wav', 'comparand_drums_bad.wav'): 170.52499999999998
		, ('reference_drums.wav', 'comparand_drums_good.wav'): 166.9075
		, ('reference_other.wav', 'comparand_other_bad.wav'): 170.76
		, ('reference_other.wav', 'comparand_other_good.wav'): 167.481
		, ('reference_vocals.wav', 'comparand_vocals_bad.wav'): 170.31150000000002
		, ('reference_vocals.wav', 'comparand_vocals_good.wav'): 167.747
	}
	, 'analyzeSDRmean': {
		('reference_bass.wav', 'comparand_bass_bad.wav'): 5.997285
		, ('reference_bass.wav', 'comparand_bass_good.wav'): 20.1554
		, ('reference_drums.wav', 'comparand_drums_bad.wav'): 5.17577
		, ('reference_drums.wav', 'comparand_drums_good.wav'): 20.8855
		, ('reference_other.wav', 'comparand_other_bad.wav'): 4.891185
		, ('reference_other.wav', 'comparand_other_good.wav'): 19.1296
		, ('reference_vocals.wav', 'comparand_vocals_bad.wav'): 3.2346649999999997
		, ('reference_vocals.wav', 'comparand_vocals_good.wav'): 14.3705
	}
	, 'analyzeSI_SDRmean': {
		('reference_bass.wav', 'comparand_bass_bad.wav'): 9.397305
		, ('reference_bass.wav', 'comparand_bass_good.wav'): 24.65985
		, ('reference_drums.wav', 'comparand_drums_bad.wav'): 4.03173
		, ('reference_drums.wav', 'comparand_drums_good.wav'): 21.0261
		, ('reference_other.wav', 'comparand_other_bad.wav'): 3.75782
		, ('reference_other.wav', 'comparand_other_good.wav'): 19.97305
		, ('reference_vocals.wav', 'comparand_vocals_bad.wav'): 2.38279
		, ('reference_vocals.wav', 'comparand_vocals_good.wav'): 14.581900000000001
	}
	, 'analyzePSNRmeanK': {
		('reference_bass.wav', 'comparand_bass_bad.wav'): 100.0
		, ('reference_bass.wav', 'comparand_bass_good.wav'): 100.0
		, ('reference_drums.wav', 'comparand_drums_bad.wav'): 100.0
		, ('reference_drums.wav', 'comparand_drums_good.wav'): 100.0
		, ('reference_other.wav', 'comparand_other_bad.wav'): 100.0
		, ('reference_other.wav', 'comparand_other_good.wav'): 100.0
		, ('reference_vocals.wav', 'comparand_vocals_bad.wav'): 100.0
		, ('reference_vocals.wav', 'comparand_vocals_good.wav'): 100.0
	}
	, 'analyzeSDRmeanK': {
		('reference_bass.wav', 'comparand_bass_bad.wav'): 93.05412206250546
		, ('reference_bass.wav', 'comparand_bass_good.wav'): 100.0
		, ('reference_drums.wav', 'comparand_drums_bad.wav'): 91.42696434397065
		, ('reference_drums.wav', 'comparand_drums_good.wav'): 100.0
		, ('reference_other.wav', 'comparand_other_bad.wav'): 90.84395367572644
		, ('reference_other.wav', 'comparand_other_good.wav'): 100.0
		, ('reference_vocals.wav', 'comparand_vocals_bad.wav'): 87.22813647235974
		, ('reference_vocals.wav', 'comparand_vocals_good.wav'): 100.0
	}
	, 'analyzeSI_SDRmeanK': {
		('reference_bass.wav', 'comparand_bass_bad.wav'): 99.0435395357055
		, ('reference_bass.wav', 'comparand_bass_good.wav'): 100.0
		, ('reference_drums.wav', 'comparand_drums_bad.wav'): 89.01768194208685
		, ('reference_drums.wav', 'comparand_drums_good.wav'): 100.0
		, ('reference_other.wav', 'comparand_other_bad.wav'): 88.41364013069871
		, ('reference_other.wav', 'comparand_other_good.wav'): 100.0
		, ('reference_vocals.wav', 'comparand_vocals_bad.wav'): 85.2011967788884
		, ('reference_vocals.wav', 'comparand_vocals_good.wav'): 100.0
	}
}

expectedSpectrogram: dict[str, dict[tuple[str, str], float]] = {
	'analyzeBleedlessMelDBMean': {
		('reference_bass.wav', 'comparand_bass_bad.wav'): 6.827766020951385
		, ('reference_bass.wav', 'comparand_bass_good.wav'): 8.299891417093491
		, ('reference_drums.wav', 'comparand_drums_bad.wav'): 17.37343903966205
		, ('reference_drums.wav', 'comparand_drums_good.wav'): 24.255046519466646
		, ('reference_other.wav', 'comparand_other_bad.wav'): 14.171035485010643
		, ('reference_other.wav', 'comparand_other_good.wav'): 10.849409195105817
		, ('reference_vocals.wav', 'comparand_vocals_bad.wav'): 11.262750544786806
		, ('reference_vocals.wav', 'comparand_vocals_good.wav'): 28.883942587824176
	}
	, 'analyzeFullnessMelDBMean': {
		('reference_bass.wav', 'comparand_bass_bad.wav'): 43.42789802086463
		, ('reference_bass.wav', 'comparand_bass_good.wav'): 58.89875701729359
		, ('reference_drums.wav', 'comparand_drums_bad.wav'): 25.937927058245677
		, ('reference_drums.wav', 'comparand_drums_good.wav'): 75.74080290256738
		, ('reference_other.wav', 'comparand_other_bad.wav'): 16.92461195649359
		, ('reference_other.wav', 'comparand_other_good.wav'): 60.65396594640248
		, ('reference_vocals.wav', 'comparand_vocals_bad.wav'): 11.880723001489121
		, ('reference_vocals.wav', 'comparand_vocals_good.wav'): 48.8601440187626
	}
}
