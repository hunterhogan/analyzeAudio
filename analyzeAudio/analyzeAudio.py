from analyzeAudio import analyzeListPathFilenamesAudio
from typing import List
from Z0Z_tools.Z0Z_io import getPathFilenames, dataTabularTOpathFilenameDelimited
import os
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='librosa', message='.*mel frequency basis.*')

if __name__ == '__main__':
    pathFilenameJSON = '/data/musdb18hq/train/filenamesEvaluate.json'
    inputMode = 'mask'  # 'json' or 'mask'
    pathInput = "/data/dataSynthesized"
    maskFilename = '*.wav'
    pathOutput = pathInput
    filenameOutput = 'audioAnalyzed.tab'
    pathFilenameOutput = os.path.join(pathOutput, filenameOutput)
    listPathFilenames = getPathFilenames(pathInput, maskFilename, inputMode, pathFilenameJSON)

    os.makedirs(pathOutput, exist_ok=True)
    
    listAspectsTarget: List[str] = []
    listAspectsTarget += ['Duration-samples', 'Peak dB', 
                          'RMS peak', 
                          'RMS total', 
                          'DC offset',
                          ]
    listAspectsTarget += ['Crest factor', 'Signal entropy', 'Zero-crossings rate', 'Dynamic range']
    listAspectsTarget += ['Abs_Peak_count', 'Bit_depth', 'Flat_factor', 'Max_difference', 'Max_level', 'Mean_difference', 'Min_difference', 'Min_level', 'Noise_floor', 'Noise_floor_count', 'Peak_count', 'RMS_difference', 'RMS_trough',]

    listAspectsTarget += ['LUFS integrated', 'LUFS loudness range', 'LUFS low', 'LUFS high']

    listAspectsTarget += ['Spectral mean', 'Spectral variance', 'Spectral centroid', 'Spectral spread', 'Spectral skewness', 'Spectral kurtosis']
    listAspectsTarget += ['Spectral entropy', 'Spectral flatness', 'Spectral crest', 'Spectral flux', 'Spectral slope', 'Spectral decrease', 'Spectral rolloff']

    listAspectsTarget += []
    listAspectsTarget += ['SRMR mean', 'RMS mean', 'Tempogram mean', 'Tempo mean']
    listAspectsTarget += ['Spectral Centroid mean', 'Chromagram mean', 'Spectral Contrast mean', 'Spectral Flatness mean', 'Spectral Bandwidth mean']
 
    rowsListFilenameAspectValues = analyzeListPathFilenamesAudio(listPathFilenames, listAspectsTarget)

    dataTabularTOpathFilenameDelimited(pathFilenameOutput, rowsListFilenameAspectValues, tableColumns = ['filename'] + listAspectsTarget, delimiterOutput='\t')
