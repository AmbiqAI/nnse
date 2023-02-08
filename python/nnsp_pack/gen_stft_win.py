"""
short time Fourier transform window generator
"""
import numpy as np
import matplotlib.pyplot as plt
from .converter_fix_point import fakefix

def gen_stft_win(   win_size        = 240,
                    hop             = 80,
                    len_fft         = 256,
                    make_c_table    = False):
    """
    STFT window generator
    """
    hop_div_fr = hop / win_size
    indices = np.arange(win_size)
    win_square = hop_div_fr * (1 - np.cos(2 * np.pi / win_size * indices ))
    win = np.sqrt(win_square)
    win = fakefix(win, 16, 15)

    if make_c_table:
        filename='../lib_nnsp/src/window_stft_coef.c'
        with open(filename, 'w') as file:
            file.write('#include <stdint.h>\n')
            file.write('#include "ambiq_nnsp_const.h"\n')
            file.write('const int16_t len_stft_win_coeff = LEN_STFT_WIN_COEFF;\n')
            file.write('const int16_t hop = LEN_STFT_HOP;\n' )
            file.write(f'//const int16_t len_fft = {len_fft};\n' )
            file.write('const int16_t stft_win_coeff[] = {')
            for entry in win:
                file.write(f'0x{int(entry * 2**15):04X},' )
            file.write('};\n')

    return win

if __name__ == '__main__':
    WIN_SIZE = 480
    HOP = 160
    LEN_FFT = 512
    window = gen_stft_win(win_size=WIN_SIZE, hop=HOP, len_fft=LEN_FFT)

    y = np.zeros(HOP)
    for i in range(round(WIN_SIZE/HOP)):
        y = y + window[HOP*i : HOP * (i+1)]**2

    fig, axs = plt.subplots(2)
    axs[0].plot(window)
    axs[0].set_ylim([0,1.1])
    axs[1].plot(y)
    axs[1].set_ylim([0,1.1])
    plt.show()
