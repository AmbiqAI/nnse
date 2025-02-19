import torchaudio
from torchaudio.pipelines import SQUIM_OBJECTIVE
import torchaudio.functional as F

folder = 'test_results/config_unet_relu_large_noncausal/keyboard_steak'
y_no, fs = torchaudio.load(f'{folder}/noisy.wav')
y_en, fs = torchaudio.load(f'{folder}/enhance.wav')

if fs != 16000:
    y_no = F.resample(y_no, fs, 16000)
    y_en = F.resample(y_en, fs, 16000)

objective_model = SQUIM_OBJECTIVE.get_model()

stoi_hyp, pesq_hyp, si_sdr_hyp = objective_model(y_no)
print(folder)
print(f"---Noisy---")
print(f"STOI: {stoi_hyp}")
print(f"PESQ: {pesq_hyp}")
print(f"SI-SDR: {si_sdr_hyp}\n")

stoi_hyp, pesq_hyp, si_sdr_hyp = objective_model(y_en)
print(f"---Enhance---")
print(f"STOI: {stoi_hyp}")
print(f"PESQ: {pesq_hyp}")
print(f"SI-SDR: {si_sdr_hyp}\n")