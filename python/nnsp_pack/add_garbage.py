"""
This module deals with sythesizing target data and grabage (out-of-vocabulary) data
"""
import logging
import numpy as np
import soundfile as sf
import librosa

def get_garbage_files(fnames):
    """ Fetch all out-of-vocabulary wavefiles """
    with open(fnames, 'r') as file: # pylint: disable=W1514
        lines = file.readlines()
    fnames = [line.strip() for line in lines]
    return fnames

def get_garbage_audio(fnames):
    """Randomly pick the audio from one of fnames"""
    idx = np.random.randint(0, len(fnames))
    try:
        data, sample_rate_in = sf.read(fnames[idx])
    except: # pylint: disable=W0702
        logging.debug('reading garbage failed %s', fnames[idx])
        data = np.random.randn(16000).astype(np.float32) * 0.1
    else:
        if data.ndim > 1:
            data = data[:,0]

        if sample_rate_in > 16000:
            data = librosa.resample(
                    data,
                    orig_sr=sample_rate_in,
                    target_sr=16000)

        elif sample_rate_in < 16000:
            data = np.random.randn(16000).astype(np.float32) * 0.1
            print(f'{fnames[idx]} sampling rate < 16000')
    finally:
        amp = np.abs(data).max()
        if amp != 0:
            data = data / amp * np.random.uniform(0.05, 1)
    return data

def concat_garb(garb, trgt, stime, etime):
    """Synthesize garbage and target"""
    idx = np.random.randint(0,2)
    rand_val = np.random.uniform(0, 1)
    if rand_val < 0.05:
        garb *= 0

    if len(garb) > (6 * 16000):
        start = np.random.randint(0, len(garb) - 6 * 16000)
        garb = garb[start : start + 16000*6]

    if idx == 0:
        out = np.concatenate([trgt, garb])
    else:
        out = np.concatenate([garb, trgt])
        stime += len(garb)
        etime += len(garb)
    return out, stime, etime
