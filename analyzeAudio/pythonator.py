from collections import defaultdict
from typing import Dict, Any, Tuple, Union
import json
import numpy

def pythonizeFFprobe(FFprobeJSON_utf8: str):
    FFroot: Dict[str, Any] = json.loads(FFprobeJSON_utf8)
    Z0Z_dictionaries: Dict[str, Union[numpy.ndarray, Dict[str, numpy.ndarray]]] = {}
    if 'packets_and_frames' in FFroot: # Divide into 'packets' and 'frames'
        FFroot = defaultdict(list, FFroot)
        for packetOrFrame in FFroot['packets_and_frames']:
            if 'type' in packetOrFrame:
                FFroot[section := packetOrFrame['type'] + 's'].append(packetOrFrame)
                del FFroot[section][-1]['type']
            else:
                raise ValueError("'packets_and_frames' for the win!")
        del FFroot['packets_and_frames']

    Z0Z_register = [
        'aspectralstats',
        'astats', 
        'r128',
        'signalstats',
    ]
    if 'frames' in FFroot:
        leftCrumbs = False
        listTuplesBlackdetect = []
        for indexFrame, FFframe in enumerate(FFroot['frames']):
            if 'tags' in FFframe:
                if 'lavfi.black_start' in FFframe['tags']:
                    listTuplesBlackdetect.append(float(FFframe['tags']['lavfi.black_start']))
                    del FFframe['tags']['lavfi.black_start']
                if 'lavfi.black_end' in FFframe['tags']:
                    listTuplesBlackdetect[-1] = (listTuplesBlackdetect[-1], float(FFframe['tags']['lavfi.black_end']))
                    del FFframe['tags']['lavfi.black_end']

                # This is not the way to do it
                for keyName, keyValue in FFframe['tags'].items():
                    if 'lavfi' in (keyNameDeconstructed := keyName.split('.'))[0]:
                        channel = None
                        if (registrant := keyNameDeconstructed[1]) in Z0Z_register:
                            keyNameDeconstructed = keyNameDeconstructed[2:]
                            if keyNameDeconstructed[0].isdigit(): 
                                channel = int(keyNameDeconstructed[0])
                                keyNameDeconstructed = keyNameDeconstructed[1:]
                            statistic = '.'.join(keyNameDeconstructed)
                            if channel is None:
                                while True:
                                    try:
                                        Z0Z_dictionaries[registrant][statistic][indexFrame] = float(keyValue)
                                        break  # If successful, exit the loop
                                    except KeyError:                                    
                                        if registrant not in Z0Z_dictionaries:
                                            Z0Z_dictionaries[registrant] = {}
                                        elif statistic not in Z0Z_dictionaries[registrant]:
                                            Z0Z_dictionaries[registrant][statistic] = numpy.zeros((len(FFroot['frames'])))
                                        else:
                                            raise  # Re-raise the exception
                            else:
                                while True:
                                    try:
                                        Z0Z_dictionaries[registrant][statistic][channel - 1, indexFrame] = float(keyValue)
                                        break  # If successful, exit the loop
                                    except KeyError:
                                        if registrant not in Z0Z_dictionaries:
                                            Z0Z_dictionaries[registrant] = {}
                                        elif statistic not in Z0Z_dictionaries[registrant]:
                                                Z0Z_dictionaries[registrant][statistic] = numpy.zeros((channel, len(FFroot['frames'])))
                                        else:
                                            raise  # Re-raise the exception
                                    except IndexError:
                                        if channel > Z0Z_dictionaries[registrant][statistic].shape[0]:
                                            Z0Z_dictionaries[registrant][statistic].resize((channel, len(FFroot['frames'])))
                                        else:
                                            raise  # Re-raise the exception

                if not FFframe['tags']: # empty = False
                    del FFframe['tags']
            if FFframe:
                leftCrumbs = True
        if listTuplesBlackdetect:
            Z0Z_dictionaries['blackdetect'] = numpy.array(listTuplesBlackdetect, dtype=[('black_start', numpy.float32), ('black_end', numpy.float32)], copy=False)
    if not leftCrumbs:
        del FFroot['frames']
    return FFroot, Z0Z_dictionaries
