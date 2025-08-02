import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Parameters
fs = 16000  # Sampling rate in Hz
n_fft = 512  # FFT size
n_mels = 128  # Number of Mel bands

def melspec_gen(
        samplingRate=16000,
        n_fft = 512,
        n_mels = 32,
        thresh_mel=50):
    # Compute Mel filter bank
    mel_filters = librosa.filters.mel(
        sr=samplingRate,
        n_fft=n_fft,
        n_mels=n_mels)
    mel_filters /= np.max(mel_filters, axis=-1, keepdims=True)
    for i in range(len(mel_filters)):
        id = np.argmax(mel_filters[i])
        if id > thresh_mel:
            break
    mel_filters = mel_filters[i:]
    tmp = np.eye(1 + (n_fft >> 1))[:id]
    mel_filters = np.concatenate(
        (tmp, mel_filters),
        axis=0)

    return mel_filters

if __name__ == "__main__":
    from c_code_table_converter import float2fix
    fs = 16000  # Sampling rate in Hz
    n_fft = 512  # FFT size
    n_mels = 32  # Number of Mel bands
    thresh_mel = 50  # Threshold for Mel filter bank
    mel_filters = melspec_gen(
        fs, n_fft, n_mels, thresh_mel=thresh_mel)
    print(mel_filters.shape)
    
    with open(f"melcomb_coeff_nfilt{len(mel_filters)}_fftsize{n_fft}.c", "w") as file:
        file.write("#include <stdint.h>\n")
        file.write(f"const int16_t melcomb_coeff_nfilt{len(mel_filters)} = {len(mel_filters)};\n")
        file.write(f"const int16_t mcombfltrBank_coeff_nfilt{len(mel_filters)}_fftsize512[] = {{\n")
        for i_m, vec in enumerate(mel_filters):
            indices=np.array([],dtype=int)  # Initialize an empty array to store indices
            values=np.array([], dtype=np.float32)  # Initialize an empty array to store values
            for i, v in enumerate(vec):
                if v > 0:
                    indices = np.append(indices, i)
                    values = np.append(values, v)

            M = indices.max()
            m = indices.min()
            file.write(f"0x{m:04X}, 0x{M:04X}, ")  # Print min and max indices
            for v in values:
                # Convert float value to fixed-point representation
                val = float2fix(v, 15, 16)
                file.write(f"0x{val:04X}, ")  # Print fixed-point value
            file.write(f"// mel_{i_m}: start_bin, end_bin, values")
            file.write("\n")
        file.write(f"}};\n")
    # import pdb; pdb.set_trace()
    # Plot the filters
    plt.figure(figsize=(10, 6))
    for i in range(mel_filters.shape[0]):
        plt.plot(mel_filters[i], alpha=0.7)

    plt.xlabel("FFT Bin")
    plt.ylabel("Amplitude")
    plt.title(f"{mel_filters.shape[0]} Mel Frequency Bands")
    plt.grid()
    plt.show()