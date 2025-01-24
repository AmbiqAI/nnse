import torch
import torchaudio
from torchaudio.pipelines import SQUIM_OBJECTIVE
import torchaudio.functional as F

folder = 'test_results/config_unet_tanh_noncausal/steak_hairdryer'

y_no, fs = torchaudio.load(f'{folder}/noisy.wav')
y_en, fs = torchaudio.load(f'{folder}/enhance.wav')
# fname='steak_hairdryer.wav'
# y_no, fs = torchaudio.load(f'test_wavs/{fname}')
# y_en, fs = torchaudio.load(f'df2/enhanced_{fname}')


y = torch.concat([y_no, y_en], dim=0)

if fs != 16000:
    y = F.resample(y, fs, 16000)

objective_model = SQUIM_OBJECTIVE.get_model()

stoi_hyp, pesq_hyp, si_sdr_hyp = objective_model(y)

print(f"Estimated metrics for distorted speech are:")
print(f"[noisy, enhance]")
print(f"STOI: {stoi_hyp}")
print(f"PESQ: {pesq_hyp}")
print(f"SI-SDR: {si_sdr_hyp}\n")