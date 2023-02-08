Before download the data below, please read carefully on their licence agreements.
1. `LibriSpeech and 360-hour ASR corpus` \
LibriSpeech ASR corpus is used to make out-of-vocabulary (OOV) data and can be downloaded from  [here](https://www.openslr.org/resources/12/train-clean-360.tar.gz)  360-hour clean datasets.
Download and extract the data and put it under the folder `nnsp/python/wavs/garb/en/` as shown in Table 1.
1. `MUSAN dataset` \
The MUSAN (A Music, Speech, and Noise Corpus) dataset can be download from [here](
http://www.openslr.org/17/). 
Download and extract the data to the folder named `musan` and put it under the folder `nnsp/python/wavs/noise/` as shown in Table 1.
1. `THCHS-30 dataset` \
THCHS30 is an open Chinese speech database published by Center for Speech and Language Technology (CSLT) at Tsinghua University. You can download the data from [here](
https://www.openslr.org/resources/18/data_thchs30.tgz).
Download and extract the data to the folder named `data_thchs30` and put it under the folder `nnsp/python/wavs/garb/cn/` as shown in Table 1.
1. `WHAM dataset` \
The WHAM (The WSJ0 Hipster Ambient Mixtures) dataset can be download from [here](
https://wham.whisper.ai/). 
Download and extract the data to the folder named `musan` and put it under the folder `nnsp/python/wavs/noise/` as shown in Table 1.

```py
nnse/ # root 
    evb/ 
    ns-nnsp/  
    python/   
        wavs/
            garb/
                en/LibriSpeech/
                    train-clean-360/ # LibriSpeech 360-hour ASR corpus
                cn/data_thchs30/    # THCHS-30 dataset
            noise/musan/ # MUSAN dataset
            noise/wham_noise/ # WHAM dataset
    README.md 
```
<p align="center">
  Table 1: Illustration of NNSE
</p>
