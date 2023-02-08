"""
FFT class
"""
import numpy as np
import matplotlib.pyplot as plt
from . import feature_module
from . import gen_stft_win

class stft_class: # pylint: disable=invalid-name
    """
    fft_module
    """
    def __init__(self,
                 hop = 160,
                 fftsize=512,
                 winsize=480):
        self.hop = hop
        self.fftsize = fftsize
        self.winsize = winsize
        self.overlap_size = winsize - hop
        self.ibuf = np.zeros(winsize)
        self.obuf = np.zeros(winsize)
        self.win = gen_stft_win.gen_stft_win(winsize, hop, fftsize)

    def reset(self):
        """
        reset states
        """
        self.ibuf *= 0
        self.obuf *= 0

    def stft_frame_proc(self, data_time):
        """
        stft_frame_proc
        """
        self.ibuf[self.overlap_size:] = data_time.copy()
        data_freq = np.fft.rfft(self.ibuf * self.win, self.fftsize)
        self.ibuf[:self.overlap_size] = self.ibuf[self.hop:]
        return data_freq

    def istft_frame_proc(
            self,
            data_freq,
            tfmask = 1.0,
            min_tfmask = 0.0):
        """
        istft_frame_proc
        """
        data_freq = data_freq * np.maximum(tfmask, min_tfmask)
        data = np.fft.irfft(data_freq)[:self.winsize]
        wdata = data * self.win
        self.obuf += wdata
        odata = self.obuf[:self.hop].copy()
        self.obuf[:self.overlap_size] = self.obuf[self.hop:]
        self.obuf[self.overlap_size:] = 0
        return odata

def main():
    """
    main function to test perfect reconstruction of STFT
    """
    hop = 160
    fftsize = 512
    winsize = 480
    overlap_size = winsize - hop
    win = gen_stft_win.gen_stft_win(winsize, hop, fftsize)
    data = np.random.uniform(-1,1, 10000)
    print(data)
    data_slices = feature_module.strided_app(data, winsize, hop)
    print(data_slices)

    data_freq =np.fft.rfft(data_slices * win, fftsize) # stft domain
    print(data_freq.shape)

    data_tslices = np.fft.irfft(data_freq)[:,:winsize] * win

    obuf = np.zeros(winsize) # output buffer (overlap-and-add)
    out = []
    for data_tslice in data_tslices:
        obuf = obuf + data_tslice
        out += [obuf[:hop].copy()] # need to .copy to have deep copy
        obuf[:overlap_size] = obuf[hop:]
        obuf[overlap_size:] = 0
    out = np.array(out)
    print(out.shape)
    out = out.flatten()[overlap_size:]
    data_ref = data[:out.shape[0]]
    print(out.shape)
    print(data_ref.shape)
    print(np.max(np.abs(out-data_ref)))
    print(out)

    plt.figure(1)
    plt.plot(out)
    plt.plot(data_ref)

    plt.show()

if __name__=="__main__":
    main()
