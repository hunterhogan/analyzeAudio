# analyzeAudio

Measure one or more aspects of one or more audio files.

## Note well: FFmpeg & FFprobe binaries must be in PATH

Some options to [download FFmpeg and FFprobe](https://www.ffmpeg.org/download.html) at ffmpeg.org.

## Some ways to use this package

### Use `analyzeAudioFile` to measure one or more aspects of a single audio file

```python
from analyzeAudio import analyzeAudioFile
listAspectNames = ['LUFS integrated',
                   'RMS peak',
                   'SRMR mean',
                   'Spectral Flatness mean']
listMeasurements = analyzeAudioFile(pathFilename, listAspectNames)
```

### Use `getListAvailableAudioAspects` to get a crude list of aspects this package can measure

The aspect names are accurate, but the lack of additional documentation can make things challenging. 'Zero-crossing rate', 'Zero-crossing rate mean', and 'Zero-crossings rate', for example, are different from each other. ("... lack of additional documentation ...")

```python
import analyzeAudio
analyzeAudio.getListAvailableAudioAspects()
```

### Use `analyzeAudioListPathFilenames` to measure one or more aspects of individual file in a list of audio files

### Use `audioAspects` to call an analyzer function by using the name of the aspect you wish to measure

```python
from analyzeAudio import audioAspects
SI_SDR_channelsMean = audioAspects['SI-SDR mean']['analyzer'](pathFilenameAudioFile, pathFilenameDifferentAudioFile)
```

Retrieve the names of the parameters for an analyzer function with the `['analyzerParameters']` key-name.

```python
from analyzeAudio import audioAspects
print(audioAspects['Chromagram']['analyzerParameters'])
```

### Use `whatMeasurements` command line tool to list available measurements

```sh
(.venv) C:\apps\analyzeAudio>whatMeasurements
['Abs_Peak_count', 'Bit_depth', 'Chromagram', 'Chromagram mean', 'Crest factor', 'DC offset', 'Duration-samples', 'Dynamic range', 'Flat_factor', 'LUFS high', 'LUFS integrated', 'LUFS loudness range', 'LUFS low', 'Max_difference', 'Max_level', 'Mean_difference', 'Min_difference', 'Min_level', 'Noise_floor', 'Noise_floor_count', 'Peak dB', 'Peak_count', 'Power spectral density', 'Power spectral density mean', 'RMS from waveform', 'RMS from waveform mean', 'RMS peak', 'RMS total', 'RMS_difference', 'RMS_trough', 'SI-SDR mean', 'SRMR', 'SRMR mean', 'Signal entropy', 'Spectral Bandwidth', 'Spectral Bandwidth mean', 'Spectral Centroid', 'Spectral Centroid mean', 'Spectral Contrast', 'Spectral Contrast mean', 'Spectral Flatness', 'Spectral Flatness mean', 'Spectral centroid', 'Spectral centroid mean', 'Spectral crest', 'Spectral crest mean', 'Spectral decrease', 'Spectral decrease mean', 'Spectral entropy', 'Spectral entropy mean', 'Spectral flatness', 'Spectral flatness mean', 'Spectral flux', 'Spectral flux mean', 'Spectral kurtosis', 'Spectral kurtosis mean', 'Spectral rolloff', 'Spectral rolloff mean', 'Spectral skewness', 'Spectral skewness mean', 'Spectral slope', 'Spectral slope mean', 'Spectral spread', 'Spectral spread mean', 'Spectral variance', 'Spectral variance mean', 'Tempo', 'Tempo mean', 'Tempogram', 'Tempogram mean', 'Zero-crossing rate', 'Zero-crossing rate mean', 'Zero-crossings rate']
```

## Some clues about the aspects

```python
'Abs_Peak_count': float
'Bit_depth': float
'Chromagram': NDArray[float64] # shape(..., 12, frames)
'Chromagram mean': float
'Crest factor': float
'DC offset': float
'Duration-samples': float
'Dynamic range': float
'Flat_factor': float
'LUFS high': float
'LUFS integrated': float
'LUFS loudness range': float
'LUFS low': float
'Max_difference': float
'Max_level': float
'Mean_difference': float
'Min_difference': float
'Min_level': float
'Noise_floor_count': float
'Noise_floor': float
'Peak dB': float
'Peak_count': float
'Power spectral density': NDArray[float64] # shape(channels, frames)
'Power spectral density mean': float
'RMS from waveform': NDArray[float64] # shape(..., 1, frames)
'RMS from waveform mean': float
'RMS peak': float
'RMS total': float
'RMS_difference': float
'RMS_trough': float
'SI-SDR mean': float
'Signal entropy': float
'Spectral Bandwidth': NDArray[float64] # shape(..., 1, frames)
'Spectral Bandwidth mean': float
'Spectral centroid': NDArray[float64] # shape(channels, frames)
'Spectral centroid mean': float
'Spectral Centroid': NDArray[float64] # shape(..., 1, frames)
'Spectral Centroid mean': float
'Spectral Contrast': NDArray[float64] # shape(..., 7, frames)
'Spectral Contrast mean': float
'Spectral crest': NDArray[float64] # shape(channels, frames)
'Spectral crest mean': float
'Spectral decrease': NDArray[float64] # shape(channels, frames)
'Spectral decrease mean': float
'Spectral entropy': NDArray[float64] # shape(channels, frames)
'Spectral entropy mean': float
'Spectral flatness': NDArray[float64] # shape(channels, frames)
'Spectral flatness mean': float
'Spectral Flatness': NDArray[float64] # shape(..., 1, frames)
'Spectral Flatness mean': float
'Spectral flux': NDArray[float64] # shape(channels, frames)
'Spectral flux mean': float
'Spectral kurtosis': NDArray[float64] # shape(channels, frames)
'Spectral kurtosis mean': float
'Spectral rolloff': NDArray[float64] # shape(channels, frames)
'Spectral rolloff mean': float
'Spectral skewness': NDArray[float64] # shape(channels, frames)
'Spectral skewness mean': float
'Spectral slope': NDArray[float64] # shape(channels, frames)
'Spectral slope mean': float
'Spectral spread': NDArray[float64] # shape(channels, frames)
'Spectral spread mean': float
'Spectral variance': NDArray[float64] # shape(channels, frames)
'Spectral variance mean': float
'SRMR': NDArray[float64] # shape(...)
'SRMR mean': float
'Tempo': NDArray[float64] # shape(...)
'Tempo mean': float
'Tempogram': NDArray[float64] # shape(..., 384, samples)
'Tempogram mean': float
'Zero-crossing rate': NDArray[float64] # shape(..., 1, frames)
'Zero-crossing rate mean': float
'Zero-crossings rate': float
```

## Installation

```sh
pip install analyzeAudio
```

## My recovery

[![Static Badge](https://img.shields.io/badge/2011_August-Homeless_since-blue?style=flat)](https://HunterThinks.com/support)
[![YouTube Channel Subscribers](https://img.shields.io/youtube/channel/subscribers/UC3Gx7kz61009NbhpRtPP7tw)](https://www.youtube.com/@HunterHogan)

[![CC-BY-NC-4.0](https://github.com/hunterhogan/analyzeAudio/blob/main/CC-BY-NC-4.0.png)](https://creativecommons.org/licenses/by-nc/4.0/)
