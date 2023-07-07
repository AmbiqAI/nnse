"""
Download the required SE training dataset
"""
import os
import shutil
import tarfile
from zipfile import ZipFile
import requests

def se_download():
    """
    download se dataset
    """
    wavs = "wavs"
    os.makedirs(wavs, exist_ok=True)

    # LibriSpeech 360hr
    target_name = 'train-clean-360.tar.gz'
    dst_folder = f'./{wavs}/'
    url = f'https://us.openslr.org/resources/12/{target_name}'
    response = requests.get(url, stream=True)
    print(f"Downloading {url}")
    if response.status_code == 200:
        with open(f"{wavs}/{target_name}", 'wb') as file:
            file.write(response.raw.read())
    print(f"extract {target_name}")
    with tarfile.open(f"{wavs}/{target_name}") as file:
        file.extractall(dst_folder)

    # thchs30
    target_name = 'data_thchs30.tgz'
    dst_folder = f'./{wavs}/'
    url = f'https://openslr.elda.org/resources/18/{target_name}'
    response = requests.get(url, stream=True)
    print(f"Downloading {url}")
    if response.status_code == 200:
        with open(f"{wavs}/{target_name}", 'wb') as file:
            file.write(response.raw.read())
    print(f"extract {target_name}")
    with tarfile.open(f"{wavs}/{target_name}") as file:
        file.extractall(dst_folder)


    # noise dataset
    os.makedirs(f"{wavs}/noise", exist_ok=True)

    # musan
    target_name = 'musan.tar.gz'
    dst_folder = f'./{wavs}/noise/'
    url = f'https://us.openslr.org/resources/17/{target_name}'
    response = requests.get(url, stream=True)
    print(f"Downloading {url}")
    if response.status_code == 200:
        with open(f"{wavs}/{target_name}", 'wb') as file:
            file.write(response.raw.read())
    print(f"extract {target_name}")
    with tarfile.open(f"{wavs}/{target_name}") as file:
        file.extractall(dst_folder)

    # wham noise
    target_name = 'wham_noise.zip'
    dst_folder = f'./{wavs}/noise/'
    url = f'https://storage.googleapis.com/whisper-public/{target_name}'
    response = requests.get(url, stream=True)
    print(f"Downloading {url}")
    if response.status_code == 200:
        with open(f"{wavs}/{target_name}", 'wb') as file:
            file.write(response.raw.read())
    print(f"extract {target_name}")
    with ZipFile(f"{wavs}/{target_name}", 'r') as file:
        file.extractall(dst_folder)

    # ESC-50 noise
    target_name = 'master.zip'
    dst_folder = f'./{wavs}/noise/'
    url = f'https://github.com/karoldvl/ESC-50/archive/{target_name}'
    response = requests.get(url, stream=True)
    print(f"Downloading {url}")
    if response.status_code == 200:
        with open(f"{wavs}/{target_name}", 'wb') as file:
            file.write(response.raw.read())
    print(f"extract {target_name}")
    with ZipFile(f"{wavs}/{target_name}", 'r') as file:
        file.extractall(dst_folder)
    shutil.copyfile(
        'data/non_speech_esc50.csv',
        'wavs/noise/ESC-50-master/non_speech.csv')

    # rirs_noises noise
    target_name = 'rirs_noises.zip'
    dst_folder = f'./{wavs}/noise/'
    url = f'https://www.openslr.org/resources/28/{target_name}'
    response = requests.get(url, stream=True)
    print(f"Downloading {url}")
    if response.status_code == 200:
        with open(f"{wavs}/{target_name}", 'wb') as file:
            file.write(response.raw.read())
    print(f"extract {target_name}")
    with ZipFile(f"{wavs}/{target_name}", 'r') as file:
        file.extractall(dst_folder)

    # fsd50_lst = [
    #     "FSD50K.dev_audio.z01",
    #     "FSD50K.dev_audio.z02",
    #     "FSD50K.dev_audio.z03",
    #     "FSD50K.dev_audio.z04",
    #     "FSD50K.dev_audio.z05",
    #     "FSD50K.dev_audio.zip",
    # ]

    # for fname in fsd50_lst:
    #     url = f'https://zenodo.org/record/4060432/files/{fname}?download=1"'
    #     response = requests.get(url, stream=True)
    #     print(f"Downloading {url}")
    #     if response.status_code == 200:
    #         with open(fname, 'wb') as f:
    #             f.write(response.raw.read())
    # os.system("zip -s 0 FSD50K.dev_audio.zip --out unsplit.zip")
if __name__ == "__main__":
    se_download()
