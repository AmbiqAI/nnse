import numpy as np
import soundfile as sf

output_data = []
with open("data/output.txt", "r") as f:
    for line in f:
        # Split line into integers
        frame = [int(x) for x in line.strip().split()]
        output_data.append(frame)

# output_data is a list of frames, each frame is a list of 160 samples
# flatten the output_data for easier viewing
flat_output = [sample for frame in output_data for sample in frame]

np_output = np.array(flat_output) / 32768.0  # normalize to -1.0 to 1.0
sf.write('wavs/output.wav', np_output, 16000)