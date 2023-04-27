"""
Test trained NN model using wavefile as input
"""
import argparse
import os
import re
import wave
import numpy as np
import soundfile as sf
import sounddevice as sd
import librosa
from nnsp_pack.feature_module import display_stft_tfmask
from nnsp_pack.pyaudio_animation import AudioShowClass
from nnsp_pack.nn_infer import NNInferClass
from nnsp_pack.stft_module import stft_class
from data_se import params_audio as PARAM_AUDIO

SHOW_HISTOGRAM  = False
NP_INFERENCE    = False

class SeClass(NNInferClass):
    """
    Class to handle SE model
    """
    def __init__(
            self,
            nn_arch,
            epoch_loaded,
            params_audio,
            quantized=False,
            show_histogram=False,
            np_inference=False,
            feat_type = 'mel'):

        super().__init__(
            nn_arch,
            epoch_loaded,
            params_audio,
            quantized,
            show_histogram,
            np_inference,
            feat_type = feat_type)

        self.fbank_mel = np.load('fbank_mel.npy')

    def reset(self):
        """
        Reset se instance
        """
        print("Reset all the states from parents")
        super().reset()

    def blk_proc(self, data, wavefile="speech.wav"):
        """
        NN process for several frames
        """
        result_folder = 'test_results'
        _, fname = os.path.split(wavefile)
        params_audio = self.params_audio
        file = wave.open(f"{result_folder}/output_{fname}", "wb")
        file.setnchannels(2)
        file.setsampwidth(2)
        file.setframerate(params_audio['sample_rate'])

        bks = int(len(data) / params_audio['hop'])
        feats   = []
        specs   = []
        tfmasks =[]
        stft_inst = stft_class()
        stft_inst.reset()

        for i in range(bks):
            data_frame = data[i*params_audio['hop'] : (i+1) * params_audio['hop']]
            data_freq = stft_inst.stft_frame_proc(data_frame)

            if NP_INFERENCE:
                feat, spec, est = self.frame_proc_np(data_frame, return_all = True)
            else:
                feat, spec, est = self.frame_proc_tf(data_frame, return_all = True)
            tfmasks += [est]
            feats   += [feat]
            specs   += [spec]
            self.count_run = (self.count_run + 1) % self.num_dnsampl
            print(f"\rprocessing frame {i}", end='')
            out = stft_inst.istft_frame_proc(
                    data_freq,
                    tfmask = est,
                    min_tfmask = 0)
            out = np.array([data_frame, out]).T.flatten()
            out = np.floor(out * 2**15).astype(np.int16)
            file.writeframes(out.tobytes())
        print('\n', end='')
        tfmasks = np.array(tfmasks)
        feats   = np.array(feats)
        specs   = np.array(specs)

        fig_name = re.sub(r'\.wav', '.pdf', fname)
        display_stft_tfmask(
            data,
            specs.T,
            feats.T,
            tfmasks.T,
            sample_rate=params_audio['sample_rate'],
            print_name=f"{result_folder}/output_{fig_name}")
        file.close()

        data, samplerate = sf.read(f"{result_folder}/output_{fname}")
        sf.write(f"{result_folder}/enhanced_{fname}", data[:,1], samplerate)
        print(f'Check your enhanced speeech in {result_folder}/enhanced_{fname}')

def main(args):
    """main function"""
    epoch_loaded    = int(args.epoch_loaded)
    quantized       = args.quantized
    recording       = int(args.recording)
    test_wavefile   = args.test_wavefile

    if recording == 1:
        wavefile='test_wavs/speech.wav'
        AudioShowClass(
                record_seconds=10,
                wave_output_filename=wavefile,
                non_stop=False)
    else:
        wavefile = test_wavefile

    data, sample_rate = sf.read(wavefile)
    if data.ndim > 1:
        data=data[:,0]
    if sample_rate > PARAM_AUDIO['sample_rate']:
        data = librosa.resample(
                data,
                orig_sr=sample_rate,
                target_sr=PARAM_AUDIO['sample_rate'])
    sd.play(data, PARAM_AUDIO['sample_rate'])

    se_inst = SeClass(
            args.nn_arch,
            epoch_loaded,
            PARAM_AUDIO,
            quantized,
            show_histogram  = SHOW_HISTOGRAM,
            np_inference    = NP_INFERENCE,
            feat_type       = args.feat_type,
            )

    se_inst.blk_proc(data, wavefile=wavefile)

if __name__ == "__main__":

    argparser = argparse.ArgumentParser(
        description='Testing trained SE model')

    argparser.add_argument(
        '-a',
        '--nn_arch',
        default='nn_arch/def_se_nn_arch256_pspec_mse_reverb.txt',
        help='nn architecture')

    argparser.add_argument(
        '-ft',
        '--feat_type',
        default='pspec',
        help='feature type: \'mel\'or \'pspec\'')

    argparser.add_argument(
        '-r',
        '--recording',
        default = 0,
        help    = '1: recording the speech and test it, \
                   0: No recording.')

    argparser.add_argument(
        '-v',
        '--test_wavefile',
        default = 'test_wavs/i_like_steak.wav',
        help    = 'The wavfile name to be tested')

    argparser.add_argument(
        '-q',
        '--quantized',
        default = True,
        type=bool,
        help='is post quantization?')

    argparser.add_argument(
        '--epoch_loaded',
        default= 182, # 70
        help='starting epoch')

    main(argparser.parse_args())
