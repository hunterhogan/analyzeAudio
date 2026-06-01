from __future__ import annotations

from analyzeAudio import analyzersUseTensor, audioAspects
import math
import pytest
import torch

@pytest.mark.parametrize(
	('stringAspectName', 'analyzerName', 'boolNeedsSampleRate'),
	[
		('DCLoss', 'analyzeDCLoss', False),
		('ESRLoss', 'analyzeESRLoss', False),
		('LogCoshLoss', 'analyzeLogCoshLoss', False),
		('SNRLoss', 'analyzeSNRLoss', False),
		('SISDRLoss', 'analyzeSISDRLoss', False),
		('SDSDRLoss', 'analyzeSDSDRLoss', False),
		('STFTLoss', 'analyzeSTFTLoss', False),
		('MelSTFTLoss', 'analyzeMelSTFTLoss', True),
		('ChromaSTFTLoss', 'analyzeChromaSTFTLoss', True),
		('MultiResolutionSTFTLoss', 'analyzeMultiResolutionSTFTLoss', False),
		('RandomResolutionSTFTLoss', 'analyzeRandomResolutionSTFTLoss', False),
		('SumAndDifferenceSTFTLoss', 'analyzeSumAndDifferenceSTFTLoss', False),
	],
)
def test_auraloss_waveform_losses_return_registered_floats(
	tensorAudioAuralossCase: tuple[torch.Tensor, torch.Tensor, int], stringAspectName: str, analyzerName: str, boolNeedsSampleRate: bool,
) -> None:
	tensorAudioAlfa, tensorAudioBeta, sampleRate = tensorAudioAuralossCase
	analyzer = getattr(analyzersUseTensor, analyzerName)
	valueLoss = analyzer(tensorAudioAlfa, tensorAudioBeta, sampleRate) if boolNeedsSampleRate else analyzer(tensorAudioAlfa, tensorAudioBeta)
	assert isinstance(valueLoss, float), (
		f"{analyzerName} returned {type(valueLoss).__name__}, expected float for aspect {stringAspectName}."
	)
	assert math.isfinite(valueLoss), (
		f"{analyzerName} returned non-finite value {valueLoss} for aspect {stringAspectName}."
	)
	assert stringAspectName in audioAspects, (
		f"audioAspects did not register {stringAspectName}; available keys do not include the expected auraloss aspect name."
	)
	assert audioAspects[stringAspectName]['analyzer'] is analyzer, (
		f"audioAspects[{stringAspectName!r}] registered {audioAspects[stringAspectName]['analyzer']}, expected {analyzer}."
	)

@pytest.mark.parametrize(
	('dictionaryKeywordArguments', 'pytorchOnCPU', 'valueExpectedFast'),
	[
		({}, True, True),
		({'fast': False}, True, True),
		({'fast': False}, None, None),
		({'fast': True}, None, True),
	],
)
def test_analyze_srmr_forwards_fast_setting_from_keyword_or_cpu_flag(
	tensorAudioAuralossCase: tuple[torch.Tensor, torch.Tensor, int],
	monkeypatch: pytest.MonkeyPatch,
	dictionaryKeywordArguments: dict[str, bool],
	pytorchOnCPU: bool | None,
	valueExpectedFast: bool | None,
) -> None:
	tensorAudioAlfa, _, sampleRate = tensorAudioAuralossCase
	dictionaryCaptured: dict[str, object] = {}

	def stubSpeechReverberationModulationEnergyRatio(
		tensorAudio: torch.Tensor, integerSampleRate: int, **keywordArguments: object,
	) -> torch.Tensor:
		dictionaryCaptured['tensorShape'] = tuple(tensorAudio.shape)
		dictionaryCaptured['sampleRate'] = integerSampleRate
		dictionaryCaptured['keywordArguments'] = keywordArguments
		return torch.tensor([2.3, 5.7], dtype=torch.float32)

	monkeypatch.setattr(
		analyzersUseTensor,
		'speech_reverberation_modulation_energy_ratio',
		stubSpeechReverberationModulationEnergyRatio,
	)

	tensorResult = analyzersUseTensor.analyzeSRMR(
		tensorAudioAlfa,
		sampleRate,
		pytorchOnCPU=pytorchOnCPU,
		**dictionaryKeywordArguments,
	)

	assert isinstance(tensorResult, torch.Tensor), (
		f"analyzeSRMR returned {type(tensorResult).__name__}, expected Tensor for {dictionaryKeywordArguments=} and {pytorchOnCPU=}."
	)
	assert dictionaryCaptured['sampleRate'] == sampleRate, (
		f"analyzeSRMR forwarded sample rate {dictionaryCaptured['sampleRate']}, expected {sampleRate}."
	)
	assert dictionaryCaptured['keywordArguments']['fast'] is valueExpectedFast, (
		f"analyzeSRMR forwarded fast={dictionaryCaptured['keywordArguments']['fast']}, expected {valueExpectedFast} for {dictionaryKeywordArguments=} and {pytorchOnCPU=}."
	)

@pytest.mark.parametrize(
	('tensorSrmrValues', 'valueExpectedMean'),
	[
		(torch.tensor([2.0, 5.0, 11.0], dtype=torch.float32), 6.0),
	],
)
def test_analyze_srmr_mean_reduces_analyze_srmr_tensor_to_python_float(
	tensorAudioAuralossCase: tuple[torch.Tensor, torch.Tensor, int],
	monkeypatch: pytest.MonkeyPatch,
	tensorSrmrValues: torch.Tensor,
	valueExpectedMean: float,
) -> None:
	tensorAudioAlfa, _, sampleRate = tensorAudioAuralossCase

	def stubAnalyzeSrmr(
		tensorAudio: torch.Tensor, integerSampleRate: int, *, pytorchOnCPU: bool | None, **keywordArguments: object,
	) -> torch.Tensor:
		return tensorSrmrValues

	monkeypatch.setattr(analyzersUseTensor, 'analyzeSRMR', stubAnalyzeSrmr)

	valueMean = analyzersUseTensor.analyzeSRMRMean(tensorAudioAlfa, sampleRate, pytorchOnCPU=None)

	assert isinstance(valueMean, float), (
		f"analyzeSRMRMean returned {type(valueMean).__name__}, expected float."
	)
	assert math.isclose(valueMean, valueExpectedMean, rel_tol=1e-7, abs_tol=1e-7), (
		f"analyzeSRMRMean returned {valueMean}, expected {valueExpectedMean} from {tensorSrmrValues.tolist()}."
	)

@pytest.mark.parametrize(
	('integerSampleTrimmedExpected',),
	[
		(33007,),
	],
)
def test_analyze_log_wmse_mean_truncates_and_reshapes_inputs_for_upstream_call(
	tensorAudioAuralossCase: tuple[torch.Tensor, torch.Tensor, int],
	monkeypatch: pytest.MonkeyPatch,
	integerSampleTrimmedExpected: int,
) -> None:
	tensorAudioAlfa, tensorAudioBeta, sampleRate = tensorAudioAuralossCase
	tensorAudioAlfa = tensorAudioAlfa[:, :, :integerSampleTrimmedExpected]
	tensorAudioBeta = tensorAudioBeta.squeeze(0)[:, :integerSampleTrimmedExpected + 2]
	tensorAudioCharlie = tensorAudioAlfa[0, 0, :integerSampleTrimmedExpected + 4]
	dictionaryCaptured: dict[str, object] = {}

	class StubLogWMSE:
		def __init__(self, **keywordArguments: object) -> None:
			dictionaryCaptured['initKeywordArguments'] = keywordArguments

		def __call__(self, tensorAudioInput: torch.Tensor, tensorAudioEstimate: torch.Tensor, tensorAudioTarget: torch.Tensor) -> torch.Tensor:
			dictionaryCaptured['inputShape'] = tuple(tensorAudioInput.shape)
			dictionaryCaptured['estimateShape'] = tuple(tensorAudioEstimate.shape)
			dictionaryCaptured['targetShape'] = tuple(tensorAudioTarget.shape)
			return torch.tensor(8.3, dtype=torch.float32)

	monkeypatch.setattr(analyzersUseTensor.torch_log_wmse, 'LogWMSE', StubLogWMSE)

	valueLoss = analyzersUseTensor.analyzeLogWMSEMean(
		tensorAudioAlfa,
		tensorAudioBeta,
		tensorAudioCharlie,
		sampleRate,
		reduction='mean',
	)

	assert isinstance(valueLoss, float), (
		f"analyzeLogWMSEMean returned {type(valueLoss).__name__}, expected float."
	)
	assert math.isclose(valueLoss, 8.3, rel_tol=1e-7, abs_tol=1e-7), (
		f"analyzeLogWMSEMean returned {valueLoss}, expected 8.3 from stubbed upstream result."
	)
	assert dictionaryCaptured['initKeywordArguments']['sample_rate'] == sampleRate, (
		f"analyzeLogWMSEMean forwarded sample_rate={dictionaryCaptured['initKeywordArguments']['sample_rate']}, expected {sampleRate}."
	)
	assert dictionaryCaptured['initKeywordArguments']['return_as_loss'] is False, (
		f"analyzeLogWMSEMean forwarded return_as_loss={dictionaryCaptured['initKeywordArguments']['return_as_loss']}, expected False wrapper default."
	)
	assert dictionaryCaptured['inputShape'] == (1, 1, integerSampleTrimmedExpected), (
		f"analyzeLogWMSEMean forwarded input shape {dictionaryCaptured['inputShape']}, expected (1, 1, {integerSampleTrimmedExpected})."
	)
	assert dictionaryCaptured['estimateShape'] == (1, 2, 1, integerSampleTrimmedExpected), (
		f"analyzeLogWMSEMean forwarded estimate shape {dictionaryCaptured['estimateShape']}, expected (1, 2, 1, {integerSampleTrimmedExpected})."
	)
	assert dictionaryCaptured['targetShape'] == (1, 1, 2, integerSampleTrimmedExpected), (
		f"analyzeLogWMSEMean forwarded target shape {dictionaryCaptured['targetShape']}, expected (1, 1, 2, {integerSampleTrimmedExpected})."
	)

@pytest.mark.parametrize(
	('integerSampleTrimmedExpected', 'valueLossInternal', 'dictionaryKeywordArguments'),
	[
		(32977, 4.9, {'l1_weight': 0.37}),
	],
)
def test_analyze_stft_l1snrdb_mean_negates_upstream_loss_and_forwards_name(
	tensorAudioAuralossCase: tuple[torch.Tensor, torch.Tensor, int],
	monkeypatch: pytest.MonkeyPatch,
	integerSampleTrimmedExpected: int,
	valueLossInternal: float,
	dictionaryKeywordArguments: dict[str, float],
) -> None:
	tensorAudioAlfa, tensorAudioBeta, _ = tensorAudioAuralossCase
	tensorAudioAlfa = tensorAudioAlfa[:, :, :integerSampleTrimmedExpected + 17]
	tensorAudioBeta = tensorAudioBeta.squeeze(0)[:, :integerSampleTrimmedExpected]
	dictionaryCaptured: dict[str, object] = {}

	class StubSTFTL1SNRDBLoss:
		def __init__(self, stringName: str, **keywordArguments: object) -> None:
			dictionaryCaptured['name'] = stringName
			dictionaryCaptured['keywordArguments'] = keywordArguments

		def __call__(self, tensorAudioInput: torch.Tensor, tensorAudioTarget: torch.Tensor) -> torch.Tensor:
			dictionaryCaptured['inputShape'] = tuple(tensorAudioInput.shape)
			dictionaryCaptured['targetShape'] = tuple(tensorAudioTarget.shape)
			return torch.tensor(valueLossInternal, dtype=torch.float32)

	monkeypatch.setattr(analyzersUseTensor.torch_l1_snr, 'STFTL1SNRDBLoss', StubSTFTL1SNRDBLoss)

	valueLoss = analyzersUseTensor.analyzeSTFTL1SNRDBMean(tensorAudioAlfa, tensorAudioBeta, **dictionaryKeywordArguments)

	assert math.isclose(valueLoss, -valueLossInternal, rel_tol=1e-7, abs_tol=1e-7), (
		f"analyzeSTFTL1SNRDBMean returned {valueLoss}, expected {-valueLossInternal} after negating upstream loss {valueLossInternal}."
	)
	assert dictionaryCaptured['name'] == 'STFTL1SNRDB', (
		f"analyzeSTFTL1SNRDBMean constructed upstream loss with name {dictionaryCaptured['name']}, expected 'STFTL1SNRDB'."
	)
	assert dictionaryCaptured['keywordArguments'] == dictionaryKeywordArguments, (
		f"analyzeSTFTL1SNRDBMean forwarded keyword arguments {dictionaryCaptured['keywordArguments']}, expected {dictionaryKeywordArguments}."
	)
	assert dictionaryCaptured['inputShape'] == (1, 1, 2, integerSampleTrimmedExpected), (
		f"analyzeSTFTL1SNRDBMean forwarded input shape {dictionaryCaptured['inputShape']}, expected (1, 1, 2, {integerSampleTrimmedExpected})."
	)
	assert dictionaryCaptured['targetShape'] == (1, 2, integerSampleTrimmedExpected), (
		f"analyzeSTFTL1SNRDBMean forwarded target shape {dictionaryCaptured['targetShape']}, expected (1, 2, {integerSampleTrimmedExpected})."
	)
