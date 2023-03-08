"""
Feature extraction module
"""
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .gen_stft_win import gen_stft_win
from .mel import gen_mel_bank
from .converter_fix_point import fakefix
from .fakefix_fft import rfft
from .log_module import log10_fakeFix

def my_colorbar(ax_handle, im=None):
    """
    make colorbar
    """
    divider = make_axes_locatable(ax_handle)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    if im:
        plt.colorbar(im, cax=cax)
    else:
        cax.axis('off')

def display_stft_all(pcm_sn_, spec_sn_, melspec_db_sn_,
                     pcm_s_, spec_s_, melspec_db_s_,
                     sample_rate, label_frame=None):
    """
    Display stft for s and s+n
    """
    pcm_sn = pcm_sn_.copy()
    spec_sn = spec_sn_.copy()
    melspec_db_sn = melspec_db_sn_.copy()
    if pcm_sn.ndim > 1:
        pcm_sn = pcm_sn[:,0]

    pcm_s = pcm_s_.copy()
    spec_s = spec_s_.copy()
    melspec_db_s = melspec_db_s_.copy()
    if pcm_s.ndim > 1:
        pcm_s = pcm_s[:,0]

    db_lim = [-50, 10]

    # Draw the spectrogram
    specs = [spec_sn, spec_s]
    texts = ['spec (s+n)', 'spec (s)']
    i = 1
    for spec, text in zip(specs, texts):
        ax_handle = plt.subplot(6,1,i)
        i += 1
        _, len0 = spec.shape
        ax_fr = np.arange(0, len0)
        spec_db_sn = 20 * np.log10(np.maximum(np.abs(spec), 10**-5))

        im = ax_handle.imshow( spec_db_sn,
                        origin  = 'lower',
                        cmap    = 'pink_r',
                        aspect  = 'auto',
                        vmin    = db_lim[0],
                        vmax    = db_lim[1],
                        extent  = [0 , ax_fr.max(), 0 , sample_rate >> 1])

        if label_frame is not None:
            ax_handle.plot(label_frame * (sample_rate >> 2))
        ax_handle.set_ylim([0, sample_rate >> 1])
        my_colorbar(ax_handle, im)
        ax_handle.text(0.2, 0.2, text)

    # Draw the mel spec
    melspecs_db = [melspec_db_sn, melspec_db_s]
    texts = ['melspec (s+n)', 'melspec (s)']
    for melspec_db, text in zip(melspecs_db, texts):
        ax_handle = plt.subplot(6,1,i)
        i += 1
        len_feat, len0 = melspec_db.shape
        ax_fr = np.arange(0,len0)

        im = ax_handle.imshow( melspec_db,
                        origin  = 'lower',
                        cmap    = 'pink_r',
                        aspect  = 'auto',
                        vmin    = melspec_db.min(),
                        vmax    = melspec_db.max(),
                        extent  = [0 , ax_fr.max(), 0 , len_feat])

        ax_handle.set_ylim([0, len_feat])
        my_colorbar(ax_handle, im)
        ax_handle.text(0.2, 0.2, text)
        if label_frame is not None:
            ax_handle.plot(label_frame * 20)

    # Draw the time seq
    pcms_sn = [pcm_sn, pcm_s]
    texts = ['melspec (s+n)', 'melspec (s)']
    for pcm, text in zip(pcms_sn, texts):

        ax_handle = plt.subplot(6,1,i)
        i+=1
        ax_time = np.arange(0, pcm.size)/sample_rate
        ax_handle.plot(ax_time, pcm, linewidth=0.5)
        ax_handle.set_xlim([0, pcm.size/sample_rate])
        ax_handle.set_ylim([-1,1])
        my_colorbar(ax_handle)
        ax_handle.text(0.2, 0.2, 'sig')

    plt.show()

def display_stft_tfmask(
        pcm_,
        spec_,
        melspec_db_,
        tfmask,
        sample_rate,
        label_frame=None,
        print_name=None):
    """
    Display stft, tfmask
    """
    pcm = pcm_.copy()
    spec = spec_.copy()
    melspec_db = melspec_db_.copy()
    if pcm.ndim > 1:
        pcm = pcm[:,0]

    db_lim = [-50, 10]

    # Draw the spectrogram
    ax_handle = plt.subplot(4,1,1)
    _, len0 = spec.shape
    ax_fr = np.arange(0, len0)
    spec_db = 20 * np.log10(np.maximum(np.abs(spec), 10**-5))

    im = ax_handle.imshow( spec_db,
                    origin  = 'lower',
                    cmap    = 'pink_r',
                    aspect  = 'auto',
                    vmin    = db_lim[0],
                    vmax    = db_lim[1],
                    extent  = [0 , ax_fr.max(), 0 , sample_rate >> 1])

    if label_frame is not None:
        ax_handle.plot(label_frame * (sample_rate >> 2))
    ax_handle.set_ylim([0, sample_rate >> 1])
    my_colorbar(ax_handle, im)
    ax_handle.text(0.2, 0.2, 'spec')

    # Draw the mel spec
    ax_handle = plt.subplot(4,1,2)
    len_feat, len0 = melspec_db.shape
    ax_fr = np.arange(0,len0)

    im = ax_handle.imshow( melspec_db,
                    origin  = 'lower',
                    cmap    = 'pink_r',
                    aspect  = 'auto',
                    vmin    = melspec_db.min(),
                    vmax    = melspec_db.max(),
                    extent  = [0 , ax_fr.max(), 0 , len_feat])

    ax_handle.set_ylim([0, len_feat])
    my_colorbar(ax_handle, im)
    ax_handle.text(0.2, 0.2, 'mel_spec')
    if label_frame is not None:
        ax_handle.plot(label_frame * 20)

    # Draw the tfmask
    ax_handle = plt.subplot(4,1,3)
    len_feat, len0 = tfmask.shape
    ax_fr = np.arange(0,len0)

    im = ax_handle.imshow(
            tfmask,
            origin  = 'lower',
            cmap    = 'pink_r',
            aspect  = 'auto',
            vmin    = 0,
            vmax    = 1,
            extent  = [0 , ax_fr.max(), 0 , len_feat])

    ax_handle.set_ylim([0, len_feat])
    my_colorbar(ax_handle, im)
    ax_handle.text(0.2, 0.2, 'mask')
    if label_frame is not None:
        ax_handle.plot(label_frame * 20)

    # Draw the time seq
    ax_handle = plt.subplot(4,1,4)
    ax_time = np.arange(0, pcm.size)/sample_rate
    ax_handle.plot(ax_time, pcm, linewidth=0.5)
    ax_handle.set_xlim([0, pcm.size/sample_rate])
    ax_handle.set_ylim([-1,1])
    my_colorbar(ax_handle)
    ax_handle.text(0.2, 0.2, 'sig')
    if print_name:
        fig = plt.gcf()
        fig.set_size_inches((8.5, 11), forward=False)
        plt.savefig(print_name, format='pdf', dpi=500)
    plt.show()

def display_stft(pcm_, spec_, melspec_db_, sample_rate, label_frame=None):
    """
    Display stft
    """
    pcm = pcm_.copy()
    spec = spec_.copy()
    melspec_db = melspec_db_.copy()
    if pcm.ndim > 1:
        pcm = pcm[:,0]

    db_lim = [-50, 10]

    # Draw the spectrogram
    ax_handle = plt.subplot(3,1,1)
    _, len0 = spec.shape
    ax_fr = np.arange(0, len0)
    spec_db = 20 * np.log10(np.maximum(np.abs(spec), 10**-5))

    im = ax_handle.imshow( spec_db,
                    origin  = 'lower',
                    cmap    = 'pink_r',
                    aspect  = 'auto',
                    vmin    = db_lim[0],
                    vmax    = db_lim[1],
                    extent  = [0 , ax_fr.max(), 0 , sample_rate >> 1])

    if label_frame is not None:
        ax_handle.plot(label_frame * (sample_rate >> 2))
    ax_handle.set_ylim([0, sample_rate >> 1])
    my_colorbar(ax_handle, im)
    ax_handle.text(0.2, 0.2, 'spec')

    # Draw the mel spec
    ax_handle = plt.subplot(3,1,2)
    len_feat, len0 = melspec_db.shape
    ax_fr = np.arange(0,len0)

    im = ax_handle.imshow( melspec_db,
                    origin  = 'lower',
                    cmap    = 'pink_r',
                    aspect  = 'auto',
                    vmin    = melspec_db.min(),
                    vmax    = melspec_db.max(),
                    extent  = [0 , ax_fr.max(), 0 , len_feat])

    ax_handle.set_ylim([0, len_feat])
    my_colorbar(ax_handle, im)
    ax_handle.text(0.2, 0.2, 'mel_spec')
    if label_frame is not None:
        ax_handle.plot(label_frame * 20)
    # Draw the time seq
    ax_handle = plt.subplot(3,1,3)
    ax_time = np.arange(0, pcm.size)/sample_rate
    ax_handle.plot(ax_time, pcm, linewidth=0.5)
    ax_handle.set_xlim([0, pcm.size/sample_rate])
    ax_handle.set_ylim([-1,1])
    my_colorbar(ax_handle)
    ax_handle.text(0.2, 0.2, 'sig')
    plt.show()

def strided_app(data, len_win, hop ):  # Window len = L, Stride len/stepsize = S
    """
    Fast buffering
    """
    data0 = np.zeros(len_win - hop)
    data = np.concatenate((data0,data))
    nrows = ((data.size-len_win)//hop)+1
    n = data.strides[0]
    return np.lib.stride_tricks.as_strided(data, shape=(nrows,len_win), strides=(hop*n,n)).copy()

class FeatureClass:
    """
    Feature extraction module
    """
    def __init__(self,
                 win_size       = 480,
                 hop            = 160,
                 len_fft        = 512,
                 sample_rate    = 16000,
                 nfilters_mel   = 40,
                 fast_stft      = True,
                 make_c_table   = False):

        self.fast_stft = fast_stft
        self.win_size = win_size
        self.hop = hop
        self.len_fft = len_fft
        self.sample_rate = sample_rate
        self.nfilters_mel = nfilters_mel
        self.nbit_frac = 15
        self.make_c_table=make_c_table

        self.mel_banks = gen_mel_bank(  fftsize         = self.len_fft,
                                        nfilt           = self.nfilters_mel,
                                        sample_rate     = self.sample_rate,
                                        make_c_table    = self.make_c_table)

        self.win = gen_stft_win(self.win_size,
                                self.hop,
                                self.len_fft,
                                make_c_table=self.make_c_table)

        self.buf = np.zeros((self.win_size,))
        if make_c_table:
            with open('../lib_nnsp/inc/ambiq_nnsp_const.h', 'w') as file: # pylint: disable=unspecified-encoding
                file.write('#ifndef __AMBIQ_NNSP_CONST_H__\n')
                file.write('#define __AMBIQ_NNSP_CONST_H__\n')
                file.write(f'#define LEN_FFT_NNSP {len_fft}\n')
                file.write(f'#define LEN_STFT_WIN_COEFF {win_size}\n')
                file.write(f'#define LEN_STFT_HOP {hop}\n')
                file.write(f'#define NUM_MELBANKS {nfilters_mel}\n')
                file.write('#define NUM_FEATURE_CONTEXT 6\n')
                file.write('#define MAX_SIZE_FEATURE 50\n')
                file.write('#define DIMEMSION_FEATURE NUM_MELBANKS\n')
                file.write(f'#define SAMPLING_RATE {sample_rate}\n')
                file.write('#endif\n')

    def reset(self):
        """
        reset
        """
        self.buf *= 0

    def frame_proc(self, data):
        """
        Frame process
        """
        self.buf[:self.win_size - self.hop] = self.buf[self.hop:]
        self.buf[self.win_size - self.hop:] = data
        dframe_win = fakefix( self.buf * self.win, 16, 15)
        # spec = np.fft.rfft(dframe_win, self.len_fft)
        spec = rfft(dframe_win, self.len_fft, make_c_table=self.make_c_table)
        real_spec = fakefix( np.real(spec), 32, self.nbit_frac)
        imag_spec = fakefix( np.imag(spec), 32, self.nbit_frac)
        pspec = fakefix(real_spec**2 + imag_spec**2, 32, self.nbit_frac)
        mel_spec = np.matmul(pspec, self.mel_banks.T)
        mel_spec = fakefix(mel_spec, 32, self.nbit_frac)
        feat = log10_fakeFix(mel_spec)
        feat = fakefix(feat, 32, self.nbit_frac)
        return spec, mel_spec, feat, pspec

    def block_proc(self, data):
        """
        Block of frames processing
        """
        if self.fast_stft:
            dframe = strided_app(data, self.win_size, self.hop )
            dframe_win = fakefix(dframe * self.win, 16, 15)
            specs = np.fft.rfft(dframe_win, self.len_fft)
            real_specs = fakefix( np.real(specs), 32, self.nbit_frac)
            imag_specs = fakefix( np.imag(specs), 32, self.nbit_frac)
            pspecs = fakefix(real_specs**2 + imag_specs**2, 32, self.nbit_frac)
            mel_specs = np.matmul(pspecs, self.mel_banks.T)
            mel_specs = fakefix(mel_specs, 32, self.nbit_frac)
            feats = np.log10(np.maximum(mel_specs, 2**-self.nbit_frac))
            feats = fakefix(feats, 32, self.nbit_frac)
        else:
            blks = int(len(data) / self.hop)
            specs = []
            mel_specs = []
            feats = []
            pspecs = []
            for i in range(blks):
                print(f'\rFrame {i}/{blks} processing ...', end='')
                dframe = data[i*self.hop : (i+1) * self.hop]
                spec, mel_spec, feat, pspec = self.frame_proc(dframe)
                specs       += [spec]
                mel_specs   += [mel_spec]
                feats       += [feat]
                pspecs      += [pspec]

            specs = np.array(specs)
            mel_specs = np.array(mel_specs)
            feats = np.array(feats)
            pspecs = np.array(pspecs)
        # specs       = specs.astype(np.complex32)
        mel_specs   = mel_specs.astype(np.float32)
        feats       = feats.astype(np.float32)
        pspecs      = pspecs.astype(np.float32)

        return specs, mel_specs, feats, pspecs

if __name__ == '__main__':
    # True: Fast block processing
    # False: for verification. But slower
    FAST_STFT   = False

    # True  : make python ref data.
    # False : compare the python and C results
    MAKE_DATA   = False

    # True  : make all C tables of feature extraction.
    # False : disable making tables
    MAKE_C_TABLE     = False

    if MAKE_DATA:
        WAVEFILE ='test_wavs/audio.wav'

        speech, sample_rate = sf.read(WAVEFILE)

        feature_inst = FeatureClass(
                        win_size        = 240 << 1,
                        hop             = 80 << 1,
                        len_fft         = 256 << 1,
                        sample_rate     = sample_rate,
                        nfilters_mel    = 40,
                        fast_stft       = FAST_STFT,
                        make_c_table    = MAKE_C_TABLE)

        specs, mel_specs, feats, pspecs = feature_inst.block_proc(speech)

        with open('dump/file_spec_py', 'w') as file_spec: # pylint: disable=unspecified-encoding
            for vec in specs:
                for v in vec:
                    file_spec.write(f"{int(np.real(v*2**15))} {int(np.imag(v*2**15))} " )
                file_spec.write("\n")

        with open('dump/file_pspec_py', 'w') as file_pspec: # pylint: disable=unspecified-encoding
            for vec in pspecs:
                for v in vec:
                    file_pspec.write(f"{int(v*2**15)} " )
                file_pspec.write("\n")

        with open('dump/file_melSpec_py', 'w') as file_melSpecs: # pylint: disable=unspecified-encoding
            for vec in mel_specs:
                for v in vec:
                    file_melSpecs.write(f"{int(v*2**15)} ")
                file_melSpecs.write("\n")

        with open('dump/file_feat_py', 'w') as file_feats: # pylint: disable=unspecified-encoding
            for vec in feats:
                for v in vec:
                    file_feats.write(f"{int(v*2**15)} ")
                file_feats.write("\n")

        display_stft(speech, specs.T, feats.T, sample_rate)
        print(specs.shape)

    else:
        fname = "file_feat"
        spec_py = np.loadtxt(f'dump/{fname}_py', dtype=int)
        spec_c = np.loadtxt(f'C:\\Users\\Paul Chen\\Documents\\nnsp_sol\\nnsp_simu\\dump\\{fname}_c', dtype=int)
        err =  np.abs(spec_py - spec_c).max()
        print(f"Max error = {err}")

        ax_handle = plt.subplot(2,1,1)
        plt.title(f'Error = {err}')
        ax_handle.imshow(   spec_py.T,
                            origin      = 'lower',
                            cmap        = 'pink_r',
                            aspect      = 'auto')
        ax_handle.text(0,1, 'Python')

        ax_handle = plt.subplot(2,1,2)
        ax_handle.imshow(   spec_c.T,
                            origin      = 'lower',
                            cmap        = 'pink_r',
                            aspect      = 'auto')
        ax_handle.text(0,1, 'C')
        plt.show()
