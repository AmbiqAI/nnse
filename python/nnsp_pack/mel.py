"""Generate Mel frequency banks. We use the original source code from Haytham M. Fayek.
"https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html"

 We simply modify the coefficients of Mel freq banks to fake fixed-point.
"""
import numpy as np
import matplotlib.pyplot as plt
from converter_fix_point import fakefix

def gen_mel_bank(
                fftsize         = 512,
                nfilt           = 40,
                sample_rate     = 16000,
                make_c_table    = False):
    """
    Generate mel bank coefficients
    """
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin_mel = np.floor((fftsize + 1) * hz_points / sample_rate)
    # print(bin_mel)
    # print(len(bin_mel))
    fbank = np.zeros((nfilt, int(np.floor(fftsize / 2 + 1))))
    if make_c_table:
        filename = f'../../ns-nnsp/src/melSpec_coeff_nfilt{nfilt}.c'
        file = open(filename, 'w') # pylint: disable=unspecified-encoding
        file.write('#include <stdint.h>\n')
        file.write('#include "ambiq_nnsp_const.h"\n')
        file.write(f'const int16_t num_mfltrBank_nfilt{nfilt} = {nfilt};\n')
        file.write(f'//const int16_t num_fft = {fftsize};\n' )
        file.write(f"const int16_t mfltrBank_coeff_nfilt{nfilt}[]={{\n")

    for idx in range(1, nfilt + 1):
        f_m_minus = int(bin_mel[idx - 1])   # left
        f_m = int(bin_mel[idx])             # center
        f_m_plus = int(bin_mel[idx + 1])    # right
        # print(f"left: {f_m_minus+1}, right: {f_m_plus-1}")
        for k in range(f_m_minus, f_m):
            fbank[idx - 1, k] = (k - bin_mel[idx - 1]) / (bin_mel[idx] - bin_mel[idx - 1])
        for k in range(f_m, f_m_plus):
            fbank[idx - 1, k] = (bin_mel[idx + 1] - k) / (bin_mel[idx + 1] - bin_mel[idx])

        if make_c_table:
            file.write(f'0x{int(f_m_minus+1):04X}, 0x{int(f_m_plus-1):04X},')
            for k in range(f_m_minus+1,f_m_plus):
                tmp = fbank[idx-1, k]
                val = fakefix(tmp, 16, 15)
                val = val * (2**15)
                file.write(f'0x{int(val):04X},' )
            file.write(f'// {idx-1}\n' )
    fbank_q = fakefix(fbank, 16, 15)
    if make_c_table:
        file.write('};\n')
        file.close()
    return fbank_q

if __name__ == '__main__':
    fbanks = gen_mel_bank(  fftsize         = 512,
                            nfilt           = 72,
                            sample_rate     = 16000,
                            make_c_table    = True)
    fig = plt.figure()
    for i, bank in enumerate(fbanks):
        plt.plot(bank)
    plt.show()
