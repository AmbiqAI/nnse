"""
This module deals with sythesizing speech and noise data
"""
import os
import re
import logging
import soundfile as sf
import numpy as np
import librosa

def get_power(data):
    """Calculate power of data"""
    return np.mean(data**2)

def add_noise(data, noise, snr_db, stime, etime,
              return_all=False,
              snr_dB_improved = None,
              rir = None):
    """Synthesize noise and speech"""
    noise_reverb = 0
    if rir is not None:
        idx_late_reverb = np.minimum(
            np.where(np.abs(rir).max() == rir)[0][0] + 200,
            rir.size-1)
        if 0:
            rir_e = rir.copy()
            rir_e[idx_late_reverb:] = 0
            speech_reverb = np.convolve(data, rir,'same')
            speech = np.convolve(data, rir_e, 'same')
            noise_reverb = speech_reverb - speech
            noise = noise + noise_reverb
            target = speech
        else:
            speech_reverb = np.convolve(data, rir, 'same')
            target = speech_reverb
    else:
        target = data.copy()

    pw_target = get_power(target[stime:etime])
    pw_noise = get_power(noise)
    snr = 10**(snr_db/10)
    if pw_target != 0:
        target = target / np.sqrt(pw_target)
        noise_reverb = noise_reverb / np.sqrt(pw_target)
    if pw_noise != 0 and snr != 0:
        noise = noise / np.sqrt(pw_noise) / np.sqrt(snr)
    noise = noise + noise_reverb

    output = target + noise
    max_val = np.abs(output).max()
    prob = np.random.uniform(0.05, 0.95, 1)
    gain    = prob / (max_val + 10**-5)
    output  = output * gain
    target  = target * gain
    noise   = noise  * gain
    # if snr_dB_improved:
    #     gain0 = 10**(snr_dB_improved / 20)
    #     data = data + noise / gain0

    if return_all:
        return output, target
    else:
        return output

def get_noise_files_new(path_noise_folder):
    """Fetch all of noise files"""
    lst = []
    for root, _, files in os.walk(f'wavs/noise/{path_noise_folder}'):
        for file in files:
            if re.search(r'wav$', file):
                lst += [os.path.join(root, file.strip())]
    return lst

def get_noise_files(files_list, noise_type):
    """Fetch all of noise files"""
    lst = []
    for root, _, files in os.walk(f'wavs/noise/{noise_type}/{files_list}'):
        for file in files:
            if re.search(r'wav$', file):
                lst += [os.path.join(root, file.strip())]
    return lst

def get_noise(
        fnames,
        length = 16000 * 5,
        target_sampling_rate=16000):
    """Random pick ONE of noise from fnames"""
    len0 = len(fnames)
    rand_idx = np.random.randint(0, len0)

    try:
        noise, sample_rate_in = sf.read(fnames[rand_idx])
    except: # pylint: disable=W0702
        logging.debug('reading noise file %s failed', fnames[rand_idx] )
        noise = np.random.randn(target_sampling_rate).astype(np.float32) * 0.1
    else:
        if noise.ndim > 1:
            noise = noise[:,0]

        if sample_rate_in > target_sampling_rate:
            try:
                noise = librosa.resample(
                            noise,
                            orig_sr=sample_rate_in,
                            target_sr=target_sampling_rate)
            except: # pylint: disable=W0702
                logging.debug('resampling noise %s failed. Loading random noise',  fnames[id])
                noise = np.random.randn(length).astype(np.float32) * 0.1
        elif sample_rate_in < target_sampling_rate:
            logging.debug('reading noise file %s sampling rate < 16000', fnames[rand_idx])
            noise = np.random.randn(length).astype(np.float32) * 0.1

        if len(noise) > length:
            start = np.random.randint(0, len(noise)-length)
            noise = noise[start : start+length]
        else:
            try:
                repeats = int(np.ceil(length / len(noise)))
            except: # pylint: disable=bare-except
                print(fnames[rand_idx])
                noise = np.random.randn(length)
            else:
                noise = np.tile(noise, repeats)
                noise = noise[:length]

    return noise
