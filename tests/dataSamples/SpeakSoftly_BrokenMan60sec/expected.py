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

expectedTensorSpectrogram: dict[str, dict[tuple[str, str], float]] = {
	'analyzeComplexScaleInvariantSignalNoiseRatioMean': {
		('reference_bass.wav', 'comparand_bass_bad.wav'): 9.40150153294742
		, ('reference_bass.wav', 'comparand_bass_good.wav'): 24.68523997975782
		, ('reference_drums.wav', 'comparand_drums_bad.wav'): 4.041322407630274
		, ('reference_drums.wav', 'comparand_drums_good.wav'): 21.039971168750732
		, ('reference_other.wav', 'comparand_other_bad.wav'): 3.7286285787439426
		, ('reference_other.wav', 'comparand_other_good.wav'): 19.95769929201618
		, ('reference_vocals.wav', 'comparand_vocals_bad.wav'): 2.3727428521471943
		, ('reference_vocals.wav', 'comparand_vocals_good.wav'): 14.581540431964726
	}
	, 'analyzeComplexScaleInvariantSignalNoiseRatioLossMean': {
		('reference_bass.wav', 'comparand_bass_bad.wav'): -9.40150153294742
		, ('reference_bass.wav', 'comparand_bass_good.wav'): -24.68523997975782
		, ('reference_drums.wav', 'comparand_drums_bad.wav'): -4.041322407630274
		, ('reference_drums.wav', 'comparand_drums_good.wav'): -21.039971168750732
		, ('reference_other.wav', 'comparand_other_bad.wav'): -3.7286285787439426
		, ('reference_other.wav', 'comparand_other_good.wav'): -19.95769929201618
		, ('reference_vocals.wav', 'comparand_vocals_bad.wav'): -2.3727428521471943
		, ('reference_vocals.wav', 'comparand_vocals_good.wav'): -14.581540431964726
	}
	, 'analyzeSpectralConvergenceLossMean': {
		('reference_bass.wav', 'comparand_bass_bad.wav'): 0.49862650987447693
		, ('reference_bass.wav', 'comparand_bass_good.wav'): 0.09664411783810202
		, ('reference_drums.wav', 'comparand_drums_bad.wav'): 0.49332971700303835
		, ('reference_drums.wav', 'comparand_drums_good.wav'): 0.07464447972807674
		, ('reference_other.wav', 'comparand_other_bad.wav'): 0.5387842038117745
		, ('reference_other.wav', 'comparand_other_good.wav'): 0.09118212940465847
		, ('reference_vocals.wav', 'comparand_vocals_bad.wav'): 0.6765516349711842
		, ('reference_vocals.wav', 'comparand_vocals_good.wav'): 0.16145100004937493
	}
	, 'analyzeSTFTMagnitudeLossMean': {
		('reference_bass.wav', 'comparand_bass_bad.wav'): float('inf')
		, ('reference_bass.wav', 'comparand_bass_good.wav'): float('inf')
		, ('reference_drums.wav', 'comparand_drums_bad.wav'): 0.4184487937812349
		, ('reference_drums.wav', 'comparand_drums_good.wav'): 0.10334391376800187
		, ('reference_other.wav', 'comparand_other_bad.wav'): float('nan')
		, ('reference_other.wav', 'comparand_other_good.wav'): float('inf')
		, ('reference_vocals.wav', 'comparand_vocals_bad.wav'): 1.7201868772970421
		, ('reference_vocals.wav', 'comparand_vocals_good.wav'): 0.4204974641932537
	}
	, 'analyzeL1FrequencyLoss': {
		('reference_bass.wav', 'comparand_bass_bad.wav'): 55.935312413825024
		, ('reference_bass.wav', 'comparand_bass_good.wav'): 81.99172685081622
		, ('reference_drums.wav', 'comparand_drums_bad.wav'): 45.476420125875556
		, ('reference_drums.wav', 'comparand_drums_good.wav'): 85.33190733084992
		, ('reference_other.wav', 'comparand_other_bad.wav'): 47.31486262671765
		, ('reference_other.wav', 'comparand_other_good.wav'): 76.69247052117402
		, ('reference_vocals.wav', 'comparand_vocals_bad.wav'): 44.53239954397711
		, ('reference_vocals.wav', 'comparand_vocals_good.wav'): 82.56491685748584
	}
}

expectedTensor: dict[str, dict[tuple[str, str], float]] = {
	'analyzeLogWMSEMean': {
		('reference_bass.wav', 'comparand_bass_bad.wav'): 13.61288070678711
		, ('reference_bass.wav', 'comparand_bass_good.wav'): 22.123756408691406
		, ('reference_drums.wav', 'comparand_drums_bad.wav'): 9.543726921081543
		, ('reference_drums.wav', 'comparand_drums_good.wav'): 22.277042388916016
		, ('reference_other.wav', 'comparand_other_bad.wav'): 8.973960876464844
		, ('reference_other.wav', 'comparand_other_good.wav'): 22.580848693847656
		, ('reference_vocals.wav', 'comparand_vocals_bad.wav'): 8.364330291748047
		, ('reference_vocals.wav', 'comparand_vocals_good.wav'): 19.622365951538086
	}
	, 'analyzeL1SNRMean': {
		('reference_bass.wav', 'comparand_bass_bad.wav'): 2.195249080657959
		, ('reference_bass.wav', 'comparand_bass_good.wav'): 8.029800415039062
		, ('reference_drums.wav', 'comparand_drums_bad.wav'): 1.8739222288131714
		, ('reference_drums.wav', 'comparand_drums_good.wav'): 8.902081489562988
		, ('reference_other.wav', 'comparand_other_bad.wav'): 2.7666139602661133
		, ('reference_other.wav', 'comparand_other_good.wav'): 9.263158798217773
		, ('reference_vocals.wav', 'comparand_vocals_bad.wav'): 1.3462486267089844
		, ('reference_vocals.wav', 'comparand_vocals_good.wav'): 6.612167835235596
	}
	, 'analyzeL1SNRDBMean': {
		('reference_bass.wav', 'comparand_bass_bad.wav'): 1.116309404373169
		, ('reference_bass.wav', 'comparand_bass_good.wav'): 7.944385051727295
		, ('reference_drums.wav', 'comparand_drums_bad.wav'): 1.8646036386489868
		, ('reference_drums.wav', 'comparand_drums_good.wav'): 8.881950378417969
		, ('reference_other.wav', 'comparand_other_bad.wav'): 2.1211323738098145
		, ('reference_other.wav', 'comparand_other_good.wav'): 9.210844039916992
		, ('reference_vocals.wav', 'comparand_vocals_bad.wav'): -0.5397995710372925
		, ('reference_vocals.wav', 'comparand_vocals_good.wav'): 6.580639839172363
	}
	, 'analyzeMultiL1SNRDBMean': {
		('reference_bass.wav', 'comparand_bass_bad.wav'): 0.40856385231018066
		, ('reference_bass.wav', 'comparand_bass_good.wav'): 6.872795581817627
		, ('reference_drums.wav', 'comparand_drums_bad.wav'): 3.3875279426574707
		, ('reference_drums.wav', 'comparand_drums_good.wav'): 11.655378341674805
		, ('reference_other.wav', 'comparand_other_bad.wav'): 3.16900634765625
		, ('reference_other.wav', 'comparand_other_good.wav'): 10.141423225402832
		, ('reference_vocals.wav', 'comparand_vocals_bad.wav'): 0.3076368570327759
		, ('reference_vocals.wav', 'comparand_vocals_good.wav'): 8.315617561340332
	}
	, 'analyzeSTFTL1SNRDBMean': {
		('reference_bass.wav', 'comparand_bass_bad.wav'): -0.29918172955513
		, ('reference_bass.wav', 'comparand_bass_good.wav'): 5.801206111907959
		, ('reference_drums.wav', 'comparand_drums_bad.wav'): 4.910452365875244
		, ('reference_drums.wav', 'comparand_drums_good.wav'): 14.42880630493164
		, ('reference_other.wav', 'comparand_other_bad.wav'): 4.2168803215026855
		, ('reference_other.wav', 'comparand_other_good.wav'): 11.072002410888672
		, ('reference_vocals.wav', 'comparand_vocals_bad.wav'): 1.1550732851028442
		, ('reference_vocals.wav', 'comparand_vocals_good.wav'): 10.0505952835083
	}
	, 'analyzeDCLossMean': {
		('reference_bass.wav', 'comparand_bass_bad.wav'): 3.317744631203823e-05
		, ('reference_bass.wav', 'comparand_bass_good.wav'): 6.772154392820084e-07
		, ('reference_drums.wav', 'comparand_drums_bad.wav'): 3.677475319818768e-07
		, ('reference_drums.wav', 'comparand_drums_good.wav'): 3.226724913929502e-07
		, ('reference_other.wav', 'comparand_other_bad.wav'): 3.6089177228859626e-06
		, ('reference_other.wav', 'comparand_other_good.wav'): 7.140225299906433e-09
		, ('reference_vocals.wav', 'comparand_vocals_bad.wav'): 0.00015759942471049726
		, ('reference_vocals.wav', 'comparand_vocals_good.wav'): 6.712416844578684e-09
	}
	, 'analyzeESRLossMean': {
		('reference_bass.wav', 'comparand_bass_bad.wav'): 0.7958458065986633
		, ('reference_bass.wav', 'comparand_bass_good.wav'): 0.011416871100664139
		, ('reference_drums.wav', 'comparand_drums_bad.wav'): 0.31020841002464294
		, ('reference_drums.wav', 'comparand_drums_good.wav'): 0.00852588564157486
		, ('reference_other.wav', 'comparand_other_bad.wav'): 0.7181650400161743
		, ('reference_other.wav', 'comparand_other_good.wav'): 0.013620791956782341
		, ('reference_vocals.wav', 'comparand_vocals_bad.wav'): 2.1816067695617676
		, ('reference_vocals.wav', 'comparand_vocals_good.wav'): 0.03403443843126297
	}
	, 'analyzeLogCoshLossMean': {
		('reference_bass.wav', 'comparand_bass_bad.wav'): 0.0009930550586432219
		, ('reference_bass.wav', 'comparand_bass_good.wav'): 3.8210149796213955e-05
		, ('reference_drums.wav', 'comparand_drums_bad.wav'): 0.0004933620220981538
		, ('reference_drums.wav', 'comparand_drums_good.wav'): 1.325309494859539e-05
		, ('reference_other.wav', 'comparand_other_bad.wav'): 0.0006238287314772606
		, ('reference_other.wav', 'comparand_other_good.wav'): 2.3529186364612542e-05
		, ('reference_vocals.wav', 'comparand_vocals_bad.wav'): 0.0003983236674685031
		, ('reference_vocals.wav', 'comparand_vocals_good.wav'): 3.0722731025889516e-05
	}
	, 'analyzeSNRLossMean': {
		('reference_bass.wav', 'comparand_bass_bad.wav'): -0.9918630123138428
		, ('reference_bass.wav', 'comparand_bass_good.wav'): -19.424732208251953
		, ('reference_drums.wav', 'comparand_drums_bad.wav'): -5.083654880523682
		, ('reference_drums.wav', 'comparand_drums_good.wav'): -20.69384002685547
		, ('reference_other.wav', 'comparand_other_bad.wav'): -1.4382679462432861
		, ('reference_other.wav', 'comparand_other_good.wav'): -18.661483764648438
		, ('reference_vocals.wav', 'comparand_vocals_bad.wav'): 3.387661933898926
		, ('reference_vocals.wav', 'comparand_vocals_good.wav'): -14.685417175292969
	}
	, 'analyzeSISDRLossMean': {
		('reference_bass.wav', 'comparand_bass_bad.wav'): -9.397436141967773
		, ('reference_bass.wav', 'comparand_bass_good.wav'): -24.659942626953125
		, ('reference_drums.wav', 'comparand_drums_bad.wav'): -4.031733512878418
		, ('reference_drums.wav', 'comparand_drums_good.wav'): -21.026296615600586
		, ('reference_other.wav', 'comparand_other_bad.wav'): -3.757873058319092
		, ('reference_other.wav', 'comparand_other_good.wav'): -19.973011016845703
		, ('reference_vocals.wav', 'comparand_vocals_bad.wav'): -2.382457733154297
		, ('reference_vocals.wav', 'comparand_vocals_good.wav'): -14.581550598144531
	}
	, 'analyzeSDSDRLossMean': {
		('reference_bass.wav', 'comparand_bass_bad.wav'): -5.5250725746154785
		, ('reference_bass.wav', 'comparand_bass_good.wav'): -20.140735626220703
		, ('reference_drums.wav', 'comparand_drums_bad.wav'): -3.7292299270629883
		, ('reference_drums.wav', 'comparand_drums_good.wav'): -20.851482391357422
		, ('reference_other.wav', 'comparand_other_bad.wav'): -3.3635964393615723
		, ('reference_other.wav', 'comparand_other_good.wav'): -19.08609390258789
		, ('reference_vocals.wav', 'comparand_vocals_bad.wav'): -1.2540323734283447
		, ('reference_vocals.wav', 'comparand_vocals_good.wav'): -14.221287727355957
	}
	, 'analyzeSTFTLossMean': {
		('reference_bass.wav', 'comparand_bass_bad.wav'): 3.618062973022461
		, ('reference_bass.wav', 'comparand_bass_good.wav'): 1.7366775274276733
		, ('reference_drums.wav', 'comparand_drums_bad.wav'): 0.9208869934082031
		, ('reference_drums.wav', 'comparand_drums_good.wav'): 0.18262845277786255
		, ('reference_other.wav', 'comparand_other_bad.wav'): 2.750884771347046
		, ('reference_other.wav', 'comparand_other_good.wav'): 1.8986083269119263
		, ('reference_vocals.wav', 'comparand_vocals_bad.wav'): 3.1323206424713135
		, ('reference_vocals.wav', 'comparand_vocals_good.wav'): 0.5556097626686096
	}
	, 'analyzeMelSTFTLossMean': {
		('reference_bass.wav', 'comparand_bass_bad.wav'): 4.484622955322266
		, ('reference_bass.wav', 'comparand_bass_good.wav'): 2.61906099319458
		, ('reference_drums.wav', 'comparand_drums_bad.wav'): 1.1782201528549194
		, ('reference_drums.wav', 'comparand_drums_good.wav'): 0.26070767641067505
		, ('reference_other.wav', 'comparand_other_bad.wav'): 2.1551311016082764
		, ('reference_other.wav', 'comparand_other_good.wav'): 1.2134523391723633
		, ('reference_vocals.wav', 'comparand_vocals_bad.wav'): 2.774911403656006
		, ('reference_vocals.wav', 'comparand_vocals_good.wav'): 0.6293095350265503
	}
	, 'analyzeChromaSTFTLossMean': {
		('reference_bass.wav', 'comparand_bass_bad.wav'): 4.729008674621582
		, ('reference_bass.wav', 'comparand_bass_good.wav'): 3.3564698696136475
		, ('reference_drums.wav', 'comparand_drums_bad.wav'): 0.936657190322876
		, ('reference_drums.wav', 'comparand_drums_good.wav'): 0.20886445045471191
		, ('reference_other.wav', 'comparand_other_bad.wav'): 1.360734224319458
		, ('reference_other.wav', 'comparand_other_good.wav'): 0.3504577875137329
		, ('reference_vocals.wav', 'comparand_vocals_bad.wav'): 2.221552848815918
		, ('reference_vocals.wav', 'comparand_vocals_good.wav'): 0.5959670543670654
	}
	, 'analyzeMultiResolutionSTFTLossMean': {
		('reference_bass.wav', 'comparand_bass_bad.wav'): 3.4856860637664795
		, ('reference_bass.wav', 'comparand_bass_good.wav'): 1.634264349937439
		, ('reference_drums.wav', 'comparand_drums_bad.wav'): 0.9149762988090515
		, ('reference_drums.wav', 'comparand_drums_good.wav'): 0.18574325740337372
		, ('reference_other.wav', 'comparand_other_bad.wav'): 2.6518542766571045
		, ('reference_other.wav', 'comparand_other_good.wav'): 1.8173977136611938
		, ('reference_vocals.wav', 'comparand_vocals_bad.wav'): 3.0544726848602295
		, ('reference_vocals.wav', 'comparand_vocals_good.wav'): 0.5202409625053406
	}
	, 'analyzeRandomResolutionSTFTLossMean': {
		('reference_bass.wav', 'comparand_bass_bad.wav'): 3.267380952835083
		, ('reference_bass.wav', 'comparand_bass_good.wav'): 1.5489872694015503
		, ('reference_drums.wav', 'comparand_drums_bad.wav'): 1.0312467813491821
		, ('reference_drums.wav', 'comparand_drums_good.wav'): 0.2388022392988205
		, ('reference_other.wav', 'comparand_other_bad.wav'): 1.9565836191177368
		, ('reference_other.wav', 'comparand_other_good.wav'): 1.0184862613677979
		, ('reference_vocals.wav', 'comparand_vocals_bad.wav'): 2.813610315322876
		, ('reference_vocals.wav', 'comparand_vocals_good.wav'): 0.5937129855155945
	}
	, 'analyzeSumAndDifferenceSTFTLossMean': {
		('reference_bass.wav', 'comparand_bass_bad.wav'): 4.635570049285889
		, ('reference_bass.wav', 'comparand_bass_good.wav'): 2.9572482109069824
		, ('reference_drums.wav', 'comparand_drums_bad.wav'): 0.9904137849807739
		, ('reference_drums.wav', 'comparand_drums_good.wav'): 0.1620400846004486
		, ('reference_other.wav', 'comparand_other_bad.wav'): 2.763341188430786
		, ('reference_other.wav', 'comparand_other_good.wav'): 2.0389840602874756
		, ('reference_vocals.wav', 'comparand_vocals_bad.wav'): 3.230316162109375
		, ('reference_vocals.wav', 'comparand_vocals_good.wav'): 0.6018704175949097
	}
}
