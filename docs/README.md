# SE training datasets
This `README.md` gives an overview of our audio dataset we used to train the speech enhancement (SE) model. 
Before download the data below, please read carefully on their licence agreements.
The datasets will be downloaded automatically in our [python script](../python/data_se.py) when you generate the features of training data by setting the argument `download=1` as below
```python
$ python data_se.py --download=1
```
Alternatively, you can manually download those dataset.
# Outline of datasets
- `LibriSpeech and 360-hour ASR corpus` \
LibriSpeech ASR corpus is used as clean speech training dataset and can be downloaded from  [here](https://www.openslr.org/resources/12/train-clean-360.tar.gz), only 360-hour clean datasets utilized.
Download and extract the data and put it under the folder `nnsp/python/wavs/garb/en/` as shown in Table 1.
    - Citation:
        ```
        @inproceedings{panayotov2015librispeech,
            title={Librispeech: an ASR corpus based on public domain audio books},
            author={Panayotov, Vassil and Chen, Guoguo and Povey, Daniel and Khudanpur, Sanjeev},
            booktitle={Acoustics, Speech and Signal Processing (ICASSP), 2015 IEEE International Conference on},
            pages={5206--5210},
            year={2015},
            organization={IEEE}
            }
        ```
- `MUSAN dataset` \
The MUSAN (A Music, Speech, and Noise Corpus) dataset can be download from [here](
http://www.openslr.org/17/). 
Download and extract the data to the folder named `musan` and put it under the folder `nnsp/python/wavs/noise/` as shown in Table 1.
    - Citation:
        ```
        @misc{musan2015,
        author = {David Snyder and Guoguo Chen and Daniel Povey},
        title = {{MUSAN}: {A} {M}usic, {S}peech, and {N}oise {C}orpus},
        year = {2015},
        eprint = {1510.08484},
        note = {arXiv:1510.08484v1}
        }
        ```
- `THCHS-30 dataset` \
THCHS30 is an open Chinese speech database published by Center for Speech and Language Technology (CSLT) at Tsinghua University. You can download the data from [here](
https://www.openslr.org/resources/18/data_thchs30.tgz).
Download and extract the data to the folder named `data_thchs30` and put it under the folder `nnsp/python/wavs/garb/cn/` as shown in Table 1.
    - Citation:
        ```
        @misc{THCHS30_2015,
        title={THCHS-30 : A Free Chinese Speech Corpus},
        author={Dong Wang, Xuewei Zhang, Zhiyong Zhang},
        year={2015},
        url={http://arxiv.org/abs/1512.01882}
        }
        ```
- `WHAM dataset` \
The WHAM (The WSJ0 Hipster Ambient Mixtures) dataset can be download from [here](
https://wham.whisper.ai/). 
Download and extract the data to the folder named `musan` and put it under the folder `nnsp/python/wavs/noise/` as shown in Table 1.
    - Citation:
        ```
        @inproceedings{Maciejewski2020WHAMR,
            title     = {WHAMR!: Noisy and Reverberant Single-Channel Speech Separation},
            author    = {Maciejewski, Matthew and Wichern, Gordon and Le Roux, Jonathan},
            booktitle = {Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
            year      = {2020},
            month     = may
        }
        ```
- `ESC-50` \
The ESC-50 (Dataset for Environmental Sound Classification) dataset can be download from [here](
https://github.com/karoldvl/ESC-50/archive/master.zip). 
Download and extract the data to the folder named `ESC-50-master` and put it under the folder `nnsp/python/wavs/noise/` as shown in Table 1.
    - Citation:
        ```
        @inproceedings{piczak2015dataset,
        title = {{ESC}: {Dataset} for {Environmental Sound Classification}},
        author = {Piczak, Karol J.},
        booktitle = {Proceedings of the 23rd {Annual ACM Conference} on {Multimedia}},
        date = {2015-10-13},
        url = {http://dl.acm.org/citation.cfm?doid=2733373.2806390},
        doi = {10.1145/2733373.2806390},
        location = {{Brisbane, Australia}},
        isbn = {978-1-4503-3459-4},
        publisher = {{ACM Press}},
        pages = {1015--1018}
        }
        ```
```py
nnse/ # root 
    evb/ 
    ns-nnsp/  
    python/   
        wavs/
            LibriSpeech/
                train-clean-360/    # LibriSpeech 360-hour ASR corpus
            data_thchs30/           # THCHS-30 dataset
            noise/
                musan/              # MUSAN dataset
                wham_noise/         # WHAM dataset
                ESC-50-master/      # ESC-50
    README.md 
```
<p align="center">
  Table 1: Directory contents of NNSE
</p>
