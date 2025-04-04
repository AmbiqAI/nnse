import torchaudio
from torchaudio.pipelines import SQUIM_OBJECTIVE
import torchaudio.functional as F

# folder = 'test_results/config_unet_relu_noncausal_sep_specmel_th50/keyboard_steak'

def test_audio_quality(noisy, enhanced):
    """
    Test the audio quality of noisy and enhanced audio files in the given folder.
    It calculates and prints the STOI, PESQ, and SI-SDR metrics for both files.
    """

    y_no, fs = torchaudio.load(noisy)
    y_en, fs = torchaudio.load(enhanced)
    # y_en, fs = torchaudio.load(f'df2/enhanced_keyboard_steak.wav')

    if fs > 16000:
        y_no = F.resample(y_no, fs, 16000)
        y_en = F.resample(y_en, fs, 16000)
    elif fs < 16000:
        print(f"Warning: Sample rate {fs} is less than 16000 Hz. Resampling to 16000 Hz.")
        return
    objective_model = SQUIM_OBJECTIVE.get_model()

    stoi_hyp, pesq_hyp, si_sdr_hyp = objective_model(y_no)

    print(f"---Noisy---")
    print(f"STOI: {stoi_hyp}")
    print(f"PESQ: {pesq_hyp}")
    print(f"SI-SDR: {si_sdr_hyp}\n")

    stoi_hyp, pesq_hyp, si_sdr_hyp = objective_model(y_en)
    print(f"---Enhance---")
    print(f"STOI: {stoi_hyp}")
    print(f"PESQ: {pesq_hyp}")
    print(f"SI-SDR: {si_sdr_hyp}\n")