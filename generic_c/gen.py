import wave
import numpy as np

with wave.open('wavs/test.wav', 'rb') as wav:
    assert wav.getsampwidth() == 2, "WAV must be 16-bit"
    assert wav.getnchannels() == 1, "WAV must be mono"
    frames = wav.readframes(wav.getnframes())
    samples = np.frombuffer(frames, dtype=np.int16)

with open('input_data.c', 'w') as f:
    f.write('#include <stdint.h>\n')
    f.write('int16_t input_data[%d] = {\n' % len(samples))
    for i, sample in enumerate(samples):
        f.write('  %d%s' % (sample, ',' if i < len(samples)-1 else ''))
        if (i+1) % 16 == 0:
            f.write('\n')
    f.write('\n};\n')