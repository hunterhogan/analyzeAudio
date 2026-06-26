from __future__ import annotations

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
		, ('reference_bass_Hz44100.flac', 'comparand_bass_bad_Hz44100.flac'): 6.788619143331328
		, ('reference_bass_Hz44100.flac', 'comparand_bass_good_Hz44100.flac'): 8.522663704733183
		, ('reference_drums_Hz44100.flac', 'comparand_drums_bad_Hz44100.flac'): 14.913783575337419
		, ('reference_drums_Hz44100.flac', 'comparand_drums_good_Hz44100.flac'): 22.520051534306113
		, ('reference_other_Hz44100.flac', 'comparand_other_bad_Hz44100.flac'): 10.922117133776714
		, ('reference_other_Hz44100.flac', 'comparand_other_good_Hz44100.flac'): 12.113408932667035
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_bad_Hz44100.flac'): 11.443964436718353
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_good_Hz44100.flac'): 27.900681849804112
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
		, ('reference_bass_Hz44100.flac', 'comparand_bass_bad_Hz44100.flac'): 34.916990356264215
		, ('reference_bass_Hz44100.flac', 'comparand_bass_good_Hz44100.flac'): 56.43130838477892
		, ('reference_drums_Hz44100.flac', 'comparand_drums_bad_Hz44100.flac'): 24.692705961961114
		, ('reference_drums_Hz44100.flac', 'comparand_drums_good_Hz44100.flac'): 72.09360920885811
		, ('reference_other_Hz44100.flac', 'comparand_other_bad_Hz44100.flac'): 23.540335397209198
		, ('reference_other_Hz44100.flac', 'comparand_other_good_Hz44100.flac'): 68.54933372232155
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_bad_Hz44100.flac'): 12.131786864623601
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_good_Hz44100.flac'): 49.704607661368605
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
		, ('reference_bass_Hz44100.flac', 'comparand_bass_bad_Hz44100.flac'): 10.263849427690626
		, ('reference_bass_Hz44100.flac', 'comparand_bass_good_Hz44100.flac'): 25.92972068755862
		, ('reference_drums_Hz44100.flac', 'comparand_drums_bad_Hz44100.flac'): 5.710770185041959
		, ('reference_drums_Hz44100.flac', 'comparand_drums_good_Hz44100.flac'): 20.693727629539453
		, ('reference_other_Hz44100.flac', 'comparand_other_bad_Hz44100.flac'): 10.766768874275844
		, ('reference_other_Hz44100.flac', 'comparand_other_good_Hz44100.flac'): 15.578752741956539
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_bad_Hz44100.flac'): 2.3931144122167285
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_good_Hz44100.flac'): 17.125386394497212
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
		, ('reference_bass_Hz44100.flac', 'comparand_bass_bad_Hz44100.flac'): -10.263849427690626
		, ('reference_bass_Hz44100.flac', 'comparand_bass_good_Hz44100.flac'): -25.92972068755862
		, ('reference_drums_Hz44100.flac', 'comparand_drums_bad_Hz44100.flac'): -5.710770185041959
		, ('reference_drums_Hz44100.flac', 'comparand_drums_good_Hz44100.flac'): -20.693727629539453
		, ('reference_other_Hz44100.flac', 'comparand_other_bad_Hz44100.flac'): -10.766768874275844
		, ('reference_other_Hz44100.flac', 'comparand_other_good_Hz44100.flac'): -15.578752741956539
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_bad_Hz44100.flac'): -2.3931144122167285
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_good_Hz44100.flac'): -17.125386394497212
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
		, ('reference_bass_Hz44100.flac', 'comparand_bass_bad_Hz44100.flac'): 0.4875323351361945
		, ('reference_bass_Hz44100.flac', 'comparand_bass_good_Hz44100.flac'): 0.09131815796535857
		, ('reference_drums_Hz44100.flac', 'comparand_drums_bad_Hz44100.flac'): 0.4209640253592413
		, ('reference_drums_Hz44100.flac', 'comparand_drums_good_Hz44100.flac'): 0.07857831091487756
		, ('reference_other_Hz44100.flac', 'comparand_other_bad_Hz44100.flac'): 0.4219115635854795
		, ('reference_other_Hz44100.flac', 'comparand_other_good_Hz44100.flac'): 0.13357875114296142
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_bad_Hz44100.flac'): 0.6646591286245349
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_good_Hz44100.flac'): 0.1159812379418879
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
		, ('reference_bass_Hz44100.flac', 'comparand_bass_bad_Hz44100.flac'): float('inf')
		, ('reference_bass_Hz44100.flac', 'comparand_bass_good_Hz44100.flac'): float('inf')
		, ('reference_drums_Hz44100.flac', 'comparand_drums_bad_Hz44100.flac'): 0.46823404219146053
		, ('reference_drums_Hz44100.flac', 'comparand_drums_good_Hz44100.flac'): 0.10959664911230464
		, ('reference_other_Hz44100.flac', 'comparand_other_bad_Hz44100.flac'): float('inf')
		, ('reference_other_Hz44100.flac', 'comparand_other_good_Hz44100.flac'): float('inf')
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_bad_Hz44100.flac'): 1.8598160847107608
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_good_Hz44100.flac'): 0.43425770843772915
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
		, ('reference_bass_Hz44100.flac', 'comparand_bass_bad_Hz44100.flac'): 38.360092525600926
		, ('reference_bass_Hz44100.flac', 'comparand_bass_good_Hz44100.flac'): 70.59306857487417
		, ('reference_drums_Hz44100.flac', 'comparand_drums_bad_Hz44100.flac'): 47.060372647257104
		, ('reference_drums_Hz44100.flac', 'comparand_drums_good_Hz44100.flac'): 86.11919527910527
		, ('reference_other_Hz44100.flac', 'comparand_other_bad_Hz44100.flac'): 54.01775360056749
		, ('reference_other_Hz44100.flac', 'comparand_other_good_Hz44100.flac'): 76.34964459297453
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_bad_Hz44100.flac'): 47.39073500022955
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_good_Hz44100.flac'): 84.78548964039977
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
		, ('reference_bass_Hz44100.flac', 'comparand_bass_bad_Hz44100.flac'): 7.331272125244141
		, ('reference_bass_Hz44100.flac', 'comparand_bass_good_Hz44100.flac'): 16.435400009155273
		, ('reference_drums_Hz44100.flac', 'comparand_drums_bad_Hz44100.flac'): 9.276707649230957
		, ('reference_drums_Hz44100.flac', 'comparand_drums_good_Hz44100.flac'): 19.967605590820312
		, ('reference_other_Hz44100.flac', 'comparand_other_bad_Hz44100.flac'): 10.681986808776855
		, ('reference_other_Hz44100.flac', 'comparand_other_good_Hz44100.flac'): 21.390653610229492
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_bad_Hz44100.flac'): 7.1600189208984375
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_good_Hz44100.flac'): 20.05632972717285
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
		, ('reference_bass_Hz44100.flac', 'comparand_bass_bad_Hz44100.flac'): 2.688873767852783
		, ('reference_bass_Hz44100.flac', 'comparand_bass_good_Hz44100.flac'): 9.118124008178711
		, ('reference_drums_Hz44100.flac', 'comparand_drums_bad_Hz44100.flac'): 2.2721502780914307
		, ('reference_drums_Hz44100.flac', 'comparand_drums_good_Hz44100.flac'): 8.00084114074707
		, ('reference_other_Hz44100.flac', 'comparand_other_bad_Hz44100.flac'): 3.492112398147583
		, ('reference_other_Hz44100.flac', 'comparand_other_good_Hz44100.flac'): 7.7058820724487305
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_bad_Hz44100.flac'): 1.4931135177612305
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_good_Hz44100.flac'): 7.7884345054626465
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
		, ('reference_bass_Hz44100.flac', 'comparand_bass_bad_Hz44100.flac'): 1.6408628225326538
		, ('reference_bass_Hz44100.flac', 'comparand_bass_good_Hz44100.flac'): 9.03507137298584
		, ('reference_drums_Hz44100.flac', 'comparand_drums_bad_Hz44100.flac'): 2.20622181892395
		, ('reference_drums_Hz44100.flac', 'comparand_drums_good_Hz44100.flac'): 7.982356548309326
		, ('reference_other_Hz44100.flac', 'comparand_other_bad_Hz44100.flac'): 2.623063802719116
		, ('reference_other_Hz44100.flac', 'comparand_other_good_Hz44100.flac'): 7.651784420013428
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_bad_Hz44100.flac'): -0.3360309600830078
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_good_Hz44100.flac'): 7.769764423370361
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
		, ('reference_bass_Hz44100.flac', 'comparand_bass_bad_Hz44100.flac'): 1.218770980834961
		, ('reference_bass_Hz44100.flac', 'comparand_bass_good_Hz44100.flac'): 8.546062469482422
		, ('reference_drums_Hz44100.flac', 'comparand_drums_bad_Hz44100.flac'): 2.9358863830566406
		, ('reference_drums_Hz44100.flac', 'comparand_drums_good_Hz44100.flac'): 9.931452751159668
		, ('reference_other_Hz44100.flac', 'comparand_other_bad_Hz44100.flac'): 3.7261691093444824
		, ('reference_other_Hz44100.flac', 'comparand_other_good_Hz44100.flac'): 8.443642616271973
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_bad_Hz44100.flac'): 0.21733379364013672
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_good_Hz44100.flac'): 7.828123569488525
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
		, ('reference_bass_Hz44100.flac', 'comparand_bass_bad_Hz44100.flac'): 0.7966791987419128
		, ('reference_bass_Hz44100.flac', 'comparand_bass_good_Hz44100.flac'): 8.05705451965332
		, ('reference_drums_Hz44100.flac', 'comparand_drums_bad_Hz44100.flac'): 3.665550947189331
		, ('reference_drums_Hz44100.flac', 'comparand_drums_good_Hz44100.flac'): 11.880549430847168
		, ('reference_other_Hz44100.flac', 'comparand_other_bad_Hz44100.flac'): 4.8292741775512695
		, ('reference_other_Hz44100.flac', 'comparand_other_good_Hz44100.flac'): 9.23550033569336
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_bad_Hz44100.flac'): 0.7706985473632812
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_good_Hz44100.flac'): 7.8864827156066895
	}
	, 'analyzePerceptualEvaluationSpeechQualityMean': {
		('reference_bass.wav', 'comparand_bass_bad.wav'): 1.0301473140716553
		, ('reference_bass.wav', 'comparand_bass_good.wav'): 1.0711169242858887
		, ('reference_drums.wav', 'comparand_drums_bad.wav'): 1.1628379821777344
		, ('reference_drums.wav', 'comparand_drums_good.wav'): 2.8347082138061523
		, ('reference_other.wav', 'comparand_other_bad.wav'): 2.290205955505371
		, ('reference_other.wav', 'comparand_other_good.wav'): 3.148733615875244
		, ('reference_vocals.wav', 'comparand_vocals_bad.wav'): 1.3901567459106445
		, ('reference_vocals.wav', 'comparand_vocals_good.wav'): 2.1413185596466064
		, ('reference_bass_Hz44100.flac', 'comparand_bass_bad_Hz44100.flac'): 1.0286167860031128
		, ('reference_bass_Hz44100.flac', 'comparand_bass_good_Hz44100.flac'): 1.077643632888794
		, ('reference_drums_Hz44100.flac', 'comparand_drums_bad_Hz44100.flac'): 1.1721103191375732
		, ('reference_drums_Hz44100.flac', 'comparand_drums_good_Hz44100.flac'): 1.5439550876617432
		, ('reference_other_Hz44100.flac', 'comparand_other_bad_Hz44100.flac'): 2.596599578857422
		, ('reference_other_Hz44100.flac', 'comparand_other_good_Hz44100.flac'): 2.9234611988067627
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_bad_Hz44100.flac'): 1.229485034942627
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_good_Hz44100.flac'): 1.9746233224868774
	}
	, 'analyzeShortTimeObjectiveIntelligibilityMean': {
		('reference_bass.wav', 'comparand_bass_bad.wav'): 0.08338840168478226
		, ('reference_bass.wav', 'comparand_bass_good.wav'): 0.25002379348901227
		, ('reference_drums.wav', 'comparand_drums_bad.wav'): 0.6307645185871401
		, ('reference_drums.wav', 'comparand_drums_good.wav'): 0.9268120722077655
		, ('reference_other.wav', 'comparand_other_bad.wav'): 0.7319747936879211
		, ('reference_other.wav', 'comparand_other_good.wav'): 0.8701679745015911
		, ('reference_vocals.wav', 'comparand_vocals_bad.wav'): 0.4811557317310314
		, ('reference_vocals.wav', 'comparand_vocals_good.wav'): 0.7248293183321699
		, ('reference_bass_Hz44100.flac', 'comparand_bass_bad_Hz44100.flac'): 0.12635550752901004
		, ('reference_bass_Hz44100.flac', 'comparand_bass_good_Hz44100.flac'): 0.3723971618659926
		, ('reference_drums_Hz44100.flac', 'comparand_drums_bad_Hz44100.flac'): 0.6117598379774438
		, ('reference_drums_Hz44100.flac', 'comparand_drums_good_Hz44100.flac'): 0.8940359475117148
		, ('reference_other_Hz44100.flac', 'comparand_other_bad_Hz44100.flac'): 0.7396414303526102
		, ('reference_other_Hz44100.flac', 'comparand_other_good_Hz44100.flac'): 0.8788485337519831
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_bad_Hz44100.flac'): 0.5689213702733054
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_good_Hz44100.flac'): 0.7845429531111938
	}
	, 'analyzeSignalNoiseRatioMean': {
		('reference_bass.wav', 'comparand_bass_bad.wav'): 0.9917116165161133
		, ('reference_bass.wav', 'comparand_bass_good.wav'): 19.4245548248291
		, ('reference_drums.wav', 'comparand_drums_bad.wav'): 5.083650588989258
		, ('reference_drums.wav', 'comparand_drums_good.wav'): 20.693679809570312
		, ('reference_other.wav', 'comparand_other_bad.wav'): 1.438258409500122
		, ('reference_other.wav', 'comparand_other_good.wav'): 18.661479949951172
		, ('reference_vocals.wav', 'comparand_vocals_bad.wav'): -3.387763738632202
		, ('reference_vocals.wav', 'comparand_vocals_good.wav'): 14.685758590698242
		, ('reference_bass_Hz44100.flac', 'comparand_bass_bad_Hz44100.flac'): 1.159813642501831
		, ('reference_bass_Hz44100.flac', 'comparand_bass_good_Hz44100.flac'): 19.905696868896484
		, ('reference_drums_Hz44100.flac', 'comparand_drums_bad_Hz44100.flac'): 6.092227935791016
		, ('reference_drums_Hz44100.flac', 'comparand_drums_good_Hz44100.flac'): 20.38003158569336
		, ('reference_other_Hz44100.flac', 'comparand_other_bad_Hz44100.flac'): 3.2792136669158936
		, ('reference_other_Hz44100.flac', 'comparand_other_good_Hz44100.flac'): 15.029342651367188
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_bad_Hz44100.flac'): -3.000819206237793
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_good_Hz44100.flac'): 17.173877716064453
	}
	, 'analyzeScaleInvariantSignalNoiseRatioMean': {
		('reference_bass.wav', 'comparand_bass_bad.wav'): 9.397436141967773
		, ('reference_bass.wav', 'comparand_bass_good.wav'): 24.659942626953125
		, ('reference_drums.wav', 'comparand_drums_bad.wav'): 4.031733512878418
		, ('reference_drums.wav', 'comparand_drums_good.wav'): 21.026296615600586
		, ('reference_other.wav', 'comparand_other_bad.wav'): 3.757873058319092
		, ('reference_other.wav', 'comparand_other_good.wav'): 19.973011016845703
		, ('reference_vocals.wav', 'comparand_vocals_bad.wav'): 2.382457733154297
		, ('reference_vocals.wav', 'comparand_vocals_good.wav'): 14.581550598144531
		, ('reference_bass_Hz44100.flac', 'comparand_bass_bad_Hz44100.flac'): 10.250692367553711
		, ('reference_bass_Hz44100.flac', 'comparand_bass_good_Hz44100.flac'): 25.893037796020508
		, ('reference_drums_Hz44100.flac', 'comparand_drums_bad_Hz44100.flac'): 5.669340133666992
		, ('reference_drums_Hz44100.flac', 'comparand_drums_good_Hz44100.flac'): 20.645662307739258
		, ('reference_other_Hz44100.flac', 'comparand_other_bad_Hz44100.flac'): 10.775736808776855
		, ('reference_other_Hz44100.flac', 'comparand_other_good_Hz44100.flac'): 15.644216537475586
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_bad_Hz44100.flac'): 2.433537006378174
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_good_Hz44100.flac'): 17.120887756347656
	}
	, 'analyzeScaleInvariantSignalDistortionRatioMean': {
		('reference_bass.wav', 'comparand_bass_bad.wav'): 9.397308349609375
		, ('reference_bass.wav', 'comparand_bass_good.wav'): 24.65981674194336
		, ('reference_drums.wav', 'comparand_drums_bad.wav'): 4.031726837158203
		, ('reference_drums.wav', 'comparand_drums_good.wav'): 21.02611541748047
		, ('reference_other.wav', 'comparand_other_bad.wav'): 3.7578227519989014
		, ('reference_other.wav', 'comparand_other_good.wav'): 19.973007202148438
		, ('reference_vocals.wav', 'comparand_vocals_bad.wav'): 2.3827900886535645
		, ('reference_vocals.wav', 'comparand_vocals_good.wav'): 14.581903457641602
		, ('reference_bass_Hz44100.flac', 'comparand_bass_bad_Hz44100.flac'): 10.250760078430176
		, ('reference_bass_Hz44100.flac', 'comparand_bass_good_Hz44100.flac'): 25.893238067626953
		, ('reference_drums_Hz44100.flac', 'comparand_drums_bad_Hz44100.flac'): 5.669342041015625
		, ('reference_drums_Hz44100.flac', 'comparand_drums_good_Hz44100.flac'): 20.64527130126953
		, ('reference_other_Hz44100.flac', 'comparand_other_bad_Hz44100.flac'): 10.775716781616211
		, ('reference_other_Hz44100.flac', 'comparand_other_good_Hz44100.flac'): 15.644208908081055
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_bad_Hz44100.flac'): 2.434002161026001
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_good_Hz44100.flac'): 17.121379852294922
	}
	, 'analyzeSignalDistortionRatioMean': {
		('reference_bass.wav', 'comparand_bass_bad.wav'): 10.557310104370117
		, ('reference_bass.wav', 'comparand_bass_good.wav'): 26.460002899169922
		, ('reference_drums.wav', 'comparand_drums_bad.wav'): 5.965348243713379
		, ('reference_drums.wav', 'comparand_drums_good.wav'): 21.640296936035156
		, ('reference_other.wav', 'comparand_other_bad.wav'): 7.264517784118652
		, ('reference_other.wav', 'comparand_other_good.wav'): 20.351207733154297
		, ('reference_vocals.wav', 'comparand_vocals_bad.wav'): 6.0286865234375
		, ('reference_vocals.wav', 'comparand_vocals_good.wav'): 14.899955749511719
		, ('reference_bass_Hz44100.flac', 'comparand_bass_bad_Hz44100.flac'): 11.581761360168457
		, ('reference_bass_Hz44100.flac', 'comparand_bass_good_Hz44100.flac'): 27.98430061340332
		, ('reference_drums_Hz44100.flac', 'comparand_drums_bad_Hz44100.flac'): 6.709039211273193
		, ('reference_drums_Hz44100.flac', 'comparand_drums_good_Hz44100.flac'): 22.199047088623047
		, ('reference_other_Hz44100.flac', 'comparand_other_bad_Hz44100.flac'): 11.020684242248535
		, ('reference_other_Hz44100.flac', 'comparand_other_good_Hz44100.flac'): 16.045215606689453
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_bad_Hz44100.flac'): 8.276033401489258
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_good_Hz44100.flac'): 18.299453735351562
	}
	, 'analyzeSourceAggregatedSignalDistortionRatioMean': {
		('reference_bass.wav', 'comparand_bass_bad.wav'): 9.397294998168945
		, ('reference_bass.wav', 'comparand_bass_good.wav'): 24.659507751464844
		, ('reference_drums.wav', 'comparand_drums_bad.wav'): 4.028922080993652
		, ('reference_drums.wav', 'comparand_drums_good.wav'): 21.026697158813477
		, ('reference_other.wav', 'comparand_other_bad.wav'): 3.7699639797210693
		, ('reference_other.wav', 'comparand_other_good.wav'): 19.98312759399414
		, ('reference_vocals.wav', 'comparand_vocals_bad.wav'): 2.382047653198242
		, ('reference_vocals.wav', 'comparand_vocals_good.wav'): 14.575895309448242
		, ('reference_bass_Hz44100.flac', 'comparand_bass_bad_Hz44100.flac'): 10.250655174255371
		, ('reference_bass_Hz44100.flac', 'comparand_bass_good_Hz44100.flac'): 25.89249038696289
		, ('reference_drums_Hz44100.flac', 'comparand_drums_bad_Hz44100.flac'): 5.677895545959473
		, ('reference_drums_Hz44100.flac', 'comparand_drums_good_Hz44100.flac'): 20.654523849487305
		, ('reference_other_Hz44100.flac', 'comparand_other_bad_Hz44100.flac'): 10.773826599121094
		, ('reference_other_Hz44100.flac', 'comparand_other_good_Hz44100.flac'): 15.638962745666504
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_bad_Hz44100.flac'): 2.4270763397216797
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_good_Hz44100.flac'): 17.114688873291016
	}
	, 'analyzePermutationInvariantTrainingMean': {
		('reference_bass.wav', 'comparand_bass_bad.wav'): 9.397308349609375
		, ('reference_bass.wav', 'comparand_bass_good.wav'): 24.65981674194336
		, ('reference_drums.wav', 'comparand_drums_bad.wav'): 4.0317277908325195
		, ('reference_drums.wav', 'comparand_drums_good.wav'): 21.02611541748047
		, ('reference_other.wav', 'comparand_other_bad.wav'): 3.7578234672546387
		, ('reference_other.wav', 'comparand_other_good.wav'): 19.973007202148438
		, ('reference_vocals.wav', 'comparand_vocals_bad.wav'): 2.382791519165039
		, ('reference_vocals.wav', 'comparand_vocals_good.wav'): 14.581904411315918
		, ('reference_bass_Hz44100.flac', 'comparand_bass_bad_Hz44100.flac'): 10.250758171081543
		, ('reference_bass_Hz44100.flac', 'comparand_bass_good_Hz44100.flac'): 25.893239974975586
		, ('reference_drums_Hz44100.flac', 'comparand_drums_bad_Hz44100.flac'): 5.669342994689941
		, ('reference_drums_Hz44100.flac', 'comparand_drums_good_Hz44100.flac'): 20.64527130126953
		, ('reference_other_Hz44100.flac', 'comparand_other_bad_Hz44100.flac'): 10.775716781616211
		, ('reference_other_Hz44100.flac', 'comparand_other_good_Hz44100.flac'): 15.644209861755371
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_bad_Hz44100.flac'): 2.4340033531188965
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_good_Hz44100.flac'): 17.121379852294922
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
		, ('reference_bass_Hz44100.flac', 'comparand_bass_bad_Hz44100.flac'): 7.837673183530569e-05
		, ('reference_bass_Hz44100.flac', 'comparand_bass_good_Hz44100.flac'): 1.0452351943968097e-06
		, ('reference_drums_Hz44100.flac', 'comparand_drums_bad_Hz44100.flac'): 6.919915307435076e-09
		, ('reference_drums_Hz44100.flac', 'comparand_drums_good_Hz44100.flac'): 7.604963911944651e-07
		, ('reference_other_Hz44100.flac', 'comparand_other_bad_Hz44100.flac'): 2.8181761990708765e-07
		, ('reference_other_Hz44100.flac', 'comparand_other_good_Hz44100.flac'): 3.641905976792259e-08
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_bad_Hz44100.flac'): 0.00019120211072731763
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_good_Hz44100.flac'): 1.6794041712842045e-08
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
		, ('reference_bass_Hz44100.flac', 'comparand_bass_bad_Hz44100.flac'): 0.7656294703483582
		, ('reference_bass_Hz44100.flac', 'comparand_bass_good_Hz44100.flac'): 0.010219602845609188
		, ('reference_drums_Hz44100.flac', 'comparand_drums_bad_Hz44100.flac'): 0.24609585106372833
		, ('reference_drums_Hz44100.flac', 'comparand_drums_good_Hz44100.flac'): 0.00918879359960556
		, ('reference_other_Hz44100.flac', 'comparand_other_bad_Hz44100.flac'): 0.47025179862976074
		, ('reference_other_Hz44100.flac', 'comparand_other_good_Hz44100.flac'): 0.031577229499816895
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_bad_Hz44100.flac'): 1.995829463005066
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_good_Hz44100.flac'): 0.019186412915587425
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
		, ('reference_bass_Hz44100.flac', 'comparand_bass_bad_Hz44100.flac'): 0.001616592751815915
		, ('reference_bass_Hz44100.flac', 'comparand_bass_good_Hz44100.flac'): 5.8094956330023706e-05
		, ('reference_drums_Hz44100.flac', 'comparand_drums_bad_Hz44100.flac'): 0.00023800809867680073
		, ('reference_drums_Hz44100.flac', 'comparand_drums_good_Hz44100.flac'): 9.707593562779948e-06
		, ('reference_other_Hz44100.flac', 'comparand_other_bad_Hz44100.flac'): 0.00011926857405342162
		, ('reference_other_Hz44100.flac', 'comparand_other_good_Hz44100.flac'): 1.7992844732361846e-05
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_bad_Hz44100.flac'): 0.00024657428730279207
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_good_Hz44100.flac'): 1.0664638466550969e-05
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
		, ('reference_bass_Hz44100.flac', 'comparand_bass_bad_Hz44100.flac'): -1.1600607633590698
		, ('reference_bass_Hz44100.flac', 'comparand_bass_good_Hz44100.flac'): -19.905826568603516
		, ('reference_drums_Hz44100.flac', 'comparand_drums_bad_Hz44100.flac'): -6.092226982116699
		, ('reference_drums_Hz44100.flac', 'comparand_drums_good_Hz44100.flac'): -20.380380630493164
		, ('reference_other_Hz44100.flac', 'comparand_other_bad_Hz44100.flac'): -3.2792139053344727
		, ('reference_other_Hz44100.flac', 'comparand_other_good_Hz44100.flac'): -15.029346466064453
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_bad_Hz44100.flac'): 3.000699996948242
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_good_Hz44100.flac'): -17.173397064208984
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
		, ('reference_bass_Hz44100.flac', 'comparand_bass_bad_Hz44100.flac'): -10.250692367553711
		, ('reference_bass_Hz44100.flac', 'comparand_bass_good_Hz44100.flac'): -25.893037796020508
		, ('reference_drums_Hz44100.flac', 'comparand_drums_bad_Hz44100.flac'): -5.669340133666992
		, ('reference_drums_Hz44100.flac', 'comparand_drums_good_Hz44100.flac'): -20.645662307739258
		, ('reference_other_Hz44100.flac', 'comparand_other_bad_Hz44100.flac'): -10.775736808776855
		, ('reference_other_Hz44100.flac', 'comparand_other_good_Hz44100.flac'): -15.644216537475586
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_bad_Hz44100.flac'): -2.433537006378174
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_good_Hz44100.flac'): -17.120887756347656
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
		, ('reference_bass_Hz44100.flac', 'comparand_bass_bad_Hz44100.flac'): -5.777310371398926
		, ('reference_bass_Hz44100.flac', 'comparand_bass_good_Hz44100.flac'): -20.61273193359375
		, ('reference_drums_Hz44100.flac', 'comparand_drums_bad_Hz44100.flac'): -5.616349697113037
		, ('reference_drums_Hz44100.flac', 'comparand_drums_good_Hz44100.flac'): -20.519479751586914
		, ('reference_other_Hz44100.flac', 'comparand_other_bad_Hz44100.flac'): -6.943498611450195
		, ('reference_other_Hz44100.flac', 'comparand_other_good_Hz44100.flac'): -15.38676643371582
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_bad_Hz44100.flac'): -1.3749277591705322
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_good_Hz44100.flac'): -16.903228759765625
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
		, ('reference_bass_Hz44100.flac', 'comparand_bass_bad_Hz44100.flac'): 3.143754005432129
		, ('reference_bass_Hz44100.flac', 'comparand_bass_good_Hz44100.flac'): 1.3594436645507812
		, ('reference_drums_Hz44100.flac', 'comparand_drums_bad_Hz44100.flac'): 0.9528219699859619
		, ('reference_drums_Hz44100.flac', 'comparand_drums_good_Hz44100.flac'): 0.20515450835227966
		, ('reference_other_Hz44100.flac', 'comparand_other_bad_Hz44100.flac'): 2.5160017013549805
		, ('reference_other_Hz44100.flac', 'comparand_other_good_Hz44100.flac'): 2.047971487045288
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_bad_Hz44100.flac'): 3.183549165725708
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_good_Hz44100.flac'): 0.535645067691803
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
		, ('reference_bass_Hz44100.flac', 'comparand_bass_bad_Hz44100.flac'): 3.778444290161133
		, ('reference_bass_Hz44100.flac', 'comparand_bass_good_Hz44100.flac'): 1.9443919658660889
		, ('reference_drums_Hz44100.flac', 'comparand_drums_bad_Hz44100.flac'): 1.3636258840560913
		, ('reference_drums_Hz44100.flac', 'comparand_drums_good_Hz44100.flac'): 0.35470813512802124
		, ('reference_other_Hz44100.flac', 'comparand_other_bad_Hz44100.flac'): 1.9129544496536255
		, ('reference_other_Hz44100.flac', 'comparand_other_good_Hz44100.flac'): 1.1951042413711548
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_bad_Hz44100.flac'): 2.779104471206665
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_good_Hz44100.flac'): 0.653354823589325
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
		, ('reference_bass_Hz44100.flac', 'comparand_bass_bad_Hz44100.flac'): 3.5896363258361816
		, ('reference_bass_Hz44100.flac', 'comparand_bass_good_Hz44100.flac'): 2.3018958568573
		, ('reference_drums_Hz44100.flac', 'comparand_drums_bad_Hz44100.flac'): 1.317878007888794
		, ('reference_drums_Hz44100.flac', 'comparand_drums_good_Hz44100.flac'): 0.34597527980804443
		, ('reference_other_Hz44100.flac', 'comparand_other_bad_Hz44100.flac'): 1.355136513710022
		, ('reference_other_Hz44100.flac', 'comparand_other_good_Hz44100.flac'): 0.37212562561035156
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_bad_Hz44100.flac'): 2.197150230407715
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_good_Hz44100.flac'): 0.6126900911331177
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
		, ('reference_bass_Hz44100.flac', 'comparand_bass_bad_Hz44100.flac'): 3.1073949337005615
		, ('reference_bass_Hz44100.flac', 'comparand_bass_good_Hz44100.flac'): 1.3394776582717896
		, ('reference_drums_Hz44100.flac', 'comparand_drums_bad_Hz44100.flac'): 0.9509133696556091
		, ('reference_drums_Hz44100.flac', 'comparand_drums_good_Hz44100.flac'): 0.20525391399860382
		, ('reference_other_Hz44100.flac', 'comparand_other_bad_Hz44100.flac'): 2.4340732097625732
		, ('reference_other_Hz44100.flac', 'comparand_other_good_Hz44100.flac'): 1.962554931640625
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_bad_Hz44100.flac'): 3.089580774307251
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_good_Hz44100.flac'): 0.49985992908477783
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
		, ('reference_bass_Hz44100.flac', 'comparand_bass_bad_Hz44100.flac'): 2.887147903442383
		, ('reference_bass_Hz44100.flac', 'comparand_bass_good_Hz44100.flac'): 1.2316460609436035
		, ('reference_drums_Hz44100.flac', 'comparand_drums_bad_Hz44100.flac'): 1.0573173761367798
		, ('reference_drums_Hz44100.flac', 'comparand_drums_good_Hz44100.flac'): 0.28053000569343567
		, ('reference_other_Hz44100.flac', 'comparand_other_bad_Hz44100.flac'): 1.8215867280960083
		, ('reference_other_Hz44100.flac', 'comparand_other_good_Hz44100.flac'): 1.161247730255127
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_bad_Hz44100.flac'): 2.7690088748931885
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_good_Hz44100.flac'): 0.5201849341392517
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
		, ('reference_bass_Hz44100.flac', 'comparand_bass_bad_Hz44100.flac'): 4.359691619873047
		, ('reference_bass_Hz44100.flac', 'comparand_bass_good_Hz44100.flac'): 2.702949285507202
		, ('reference_drums_Hz44100.flac', 'comparand_drums_bad_Hz44100.flac'): 1.0171468257904053
		, ('reference_drums_Hz44100.flac', 'comparand_drums_good_Hz44100.flac'): 0.17078956961631775
		, ('reference_other_Hz44100.flac', 'comparand_other_bad_Hz44100.flac'): 2.5786924362182617
		, ('reference_other_Hz44100.flac', 'comparand_other_good_Hz44100.flac'): 2.1568830013275146
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_bad_Hz44100.flac'): 3.2626171112060547
		, ('reference_vocals_Hz44100.flac', 'comparand_vocals_good_Hz44100.flac'): 0.6225030422210693
	}
}
