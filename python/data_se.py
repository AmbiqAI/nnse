"""
1. Synthesize audio data
2. Feature extraction for audio data.
"""
import os
import time
import argparse
import re
import multiprocessing
import logging
import random
import numpy as np
import wandb
import boto3
import soundfile as sf
import sounddevice as sd
import librosa
from nnsp_pack import tfrecord_converter_se_split
from nnsp_pack.feature_module import FeatureClass, display_stft_all
from nnsp_pack import add_noise
from nnsp_pack import boto3_op
from nnsp_pack.se_download import se_download

DEBUG = True
UPLOAD_TFRECORD_S3 = False
DOWLOAD_DATA = False
REVERB = True
if UPLOAD_TFRECORD_S3:
    print('uploading tfrecords to s3 will slow down the process')
S3_BUCKET = "ambiqai-speech-commands-dataset"
S3_PREFIX = "tfrecords"
if DEBUG:
    SNR_DBS = [6]
else:
    SNR_DBS = [-6, -3, 0, 3, 6, 9, 12, 15, 30]

NTYPES = [
    # 'ESC-50-MASTER',
    # 'wham_noise',
    "social_noise",
    # 'FSD50K',
    # 'musan',
    # 'traffic'
]
params_audio = {
    'win_size'      : 480,
    'hop'           : 160,
    'len_fft'       : 512,
    'sample_rate'   : 16000,
    'nfilters_mel'  : 36 }

def download_data():
    """
    download data
    """
    audio_lists = [
        'data/test_files_se.csv',
        'data/train_files_se.csv',
        'data/noise_list.csv']
    s3 = boto3.client('s3')
    boto3_op.s3_download(S3_BUCKET, audio_lists)
    return s3

class FeatMultiProcsClass(multiprocessing.Process):
    """
    FeatMultiProcsClass use multiprocesses
    to run several processes of feature extraction in parallel
    """
    def __init__(self, id_process,
                 name, src_list, train_set, ntype,
                 noise_files, snr_dbs, success_dict,
                 params_audio_def,
                 num_procs = 8,
                 reverb_lst=None,
                 reverb_prob=0):

        multiprocessing.Process.__init__(self)
        self.success_dict = success_dict
        self.id_process         = id_process
        self.name               = name
        self.src_list           = src_list
        self.params_audio_def   = params_audio_def
        self.num_procs          = num_procs
        self.reverb_lst         = reverb_lst
        self.reverb_prob        = reverb_prob
        self.feat_inst      = FeatureClass(
                                win_size        = params_audio_def['win_size'],
                                hop             = params_audio_def['hop'],
                                len_fft         = params_audio_def['len_fft'],
                                sample_rate     = params_audio_def['sample_rate'],
                                nfilters_mel    = params_audio_def['nfilters_mel'])

        self.train_set      = train_set
        self.ntype          = ntype
        self.noise_files    = noise_files
        self.snr_dbs        = snr_dbs
        self.names=[]
        if DEBUG:
            self.cnt = 0

    def run(self):
        #      threadLock.acquire()
        print("Running " + self.name)
        random.shuffle(self.src_list)
        self.convert_tfrecord(
                    self.src_list,
                    self.id_process)

    def convert_tfrecord(
            self,
            fnames,
            id_process):
        """
        convert np array to tfrecord
        """
        MAX_LEN_SP = 25 * 16000
        random.shuffle(fnames)
        for i in range(len(fnames) >> 1):
            if self.num_procs-1 == self.id_process:
                print(f"\rProcessing wav {i}/{len(fnames) >> 1}", end="")
            success = 1
            stimes = []
            etimes = []
            targets = []
            speech = np.empty(0)
            len_sp_last = 0
            pattern = r'(\.wav$|\.flac$)'
            for k in range(2):
                fname = fnames[2*i+k]
                wavpath = fname.strip()
                if k == 0:
                    tfrecord = re.sub(pattern, '.tfrecord', re.sub(r'wavs', S3_PREFIX, wavpath))
                try:
                    audio, sample_rate = sf.read(wavpath)
                except :# pylint: disable=bare-except
                    success = 0
                    logging.debug("Reading the %s fails ", wavpath)
                else:
                    if audio.ndim > 1:
                        audio=audio[:,0]
                    if sample_rate > self.feat_inst.sample_rate:
                        audio = librosa.resample(
                                audio,
                                orig_sr=sample_rate,
                                target_sr=self.feat_inst.sample_rate)
                    # decorate speech
                    speech0 = audio

                    stime = np.random.randint(
                        sample_rate >> 2,
                        sample_rate << 1)
                    zeros_s = np.zeros(stime)

                    size_zeros = np.random.randint(
                        sample_rate >> 2,
                        sample_rate << 1)
                    zeros_e = np.zeros(size_zeros)
                    etime = len(speech0) + stime
                    speech0 = np.concatenate((zeros_s, speech0, zeros_e))

                    prob = np.random.uniform(0,1)
                    if prob < 0:
                        speech0 *= 0
                        target = 0
                    else:
                        target = 1
                    stimes += [stime + len_sp_last]
                    etimes += [etime + len_sp_last]
                    targets += [target]
                    speech = np.concatenate((speech, speech0))

                    len_sp_last += len(speech)
            if len(speech) > MAX_LEN_SP:
                speech = speech[:MAX_LEN_SP]
            elif len(speech) < MAX_LEN_SP:
                speech = np.pad(
                    speech,
                    (0, MAX_LEN_SP - len(speech)),
                    'constant',
                    constant_values=(0, 0))
            stimes  = np.array(stimes)
            etimes  = np.array(etimes)
            targets = np.array(targets)
            start_frames    = (stimes / self.params_audio_def['hop']) + 1 # target level frame
            start_frames    = start_frames.astype(np.int32)
            end_frames      = (etimes / self.params_audio_def['hop']) + 1 # target level frame
            end_frames      = end_frames.astype(np.int32)
            # add noise to sig
            rir = None
            reverbing = False
            if self.reverb_lst:
                rd_reverb = np.random.uniform(0,1)
                if rd_reverb < self.reverb_prob:
                    reverbing = True
                    idx = np.random.randint(0, len(self.reverb_lst))
                    # print(self.reverb_lst[idx])
                    rir, sample_rate_rir = sf.read(self.reverb_lst[idx])

                    if rir.ndim > 1:
                        rir = rir[:,0]
                    if sample_rate_rir > self.feat_inst.sample_rate:
                        rir = librosa.resample(
                                rir,
                                orig_sr=sample_rate_rir,
                                target_sr=self.feat_inst.sample_rate)
                    rir = rir[:np.minimum(3000, rir.size)]
            # add reverb
            noise = add_noise.get_noise(
                self.noise_files[self.train_set],
                len(speech),
                self.feat_inst.sample_rate)
            snr_db = self.snr_dbs[np.random.randint(0,len(self.snr_dbs))]
            audio_sn, audio_s = add_noise.add_noise(
                                    speech,
                                    noise,
                                    snr_db,
                                    stime, etime,
                                    return_all=True,
                                    snr_dB_improved = 20,
                                    rir=rir)
            # feature extraction of sig
            spec_sn, _, feat_sn, pspec_sn = self.feat_inst.block_proc(audio_sn)
            spec_s, _, feat_s, pspec_s    = self.feat_inst.block_proc(audio_s)
            if DEBUG:
                if reverbing:
                    print('has reverb')
                sd.play(
                    audio_sn,
                    self.feat_inst.sample_rate)
                print(fnames[2*i])
                print(fnames[2*i + 1])
                print(start_frames)
                print(targets)
                print(f'snr={snr_db} dB')
                flabel = np.zeros(spec_sn.shape[0])
                for start_frame, end_frame, target in zip(start_frames, end_frames, targets):
                    flabel[start_frame: end_frame] = target
                display_stft_all(audio_sn, spec_sn.T, feat_sn.T,
                                 audio_s,  spec_s.T,  feat_s.T,
                                 self.feat_inst.sample_rate,
                                 label_frame=flabel)
                os.makedirs('test_wavs', exist_ok=True)
                sf.write(f'test_wavs/speech_{self.cnt}.wav', audio_sn, self.feat_inst.sample_rate)
                sf.write(f'test_wavs/speech_{self.cnt}_ref.wav', speech, self.feat_inst.sample_rate)

                self.cnt = self.cnt + 1

            if success:
                ntype = re.sub('/','_', self.ntype)
                if reverbing:
                    tfrecord = re.sub(  r'\.tfrecord$',
                                        f'_snr{snr_db}dB_{ntype}_reverb.tfrecord',
                                        tfrecord)
                else:
                    tfrecord = re.sub(  r'\.tfrecord$',
                                        f'_snr{snr_db}dB_{ntype}.tfrecord',
                                        tfrecord)
                os.makedirs(os.path.dirname(tfrecord), exist_ok=True)
                try:
                    timesteps, _  = feat_sn.shape
                    width_targets = end_frames - start_frames + 1
                    tfrecord_converter_se_split.make_tfrecord( # pylint: disable=too-many-function-args
                                        tfrecord,
                                        pspec_sn,
                                        pspec_s)

                except: # pylint: disable=bare-except
                    print(f"Thread-{id_process}: {i}, processing {tfrecord} failed")
                else:
                    self.success_dict[self.id_process] += [tfrecord]
                    # since tfrecord file starts: data/tfrecords/speakers/...
                    # strip the leading "data/" when uploading
                    if UPLOAD_TFRECORD_S3:
                        s3.upload_file(tfrecord, S3_BUCKET, tfrecord)
                    else:
                        pass

def main(args):
    """
    main function to generate all training and testing data
    """
    download = args.download
    datasize_noise = args.datasize_noise
    reverb_prob = args.reverb_prob
    if download:
        se_download()

    if DOWLOAD_DATA:
        s3 = download_data()
    if args.wandb_track:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            job_type="data-update")
        wandb.config.update(args)

    sets_categories = ['train', 'test']

    if REVERB:
        tmp = add_noise.get_noise_files_new("rirs_noises/RIRS_NOISES/simulated_rirs")
        random.shuffle(tmp)
        start = int(len(tmp) / 5)
        lst_reverb = {}
        lst_reverb = {'train': tmp[start:],
                      'test': tmp[:start]}
    else:
        lst_reverb = None
    # Prepare noise dataset, train and test sets
    os.makedirs('data/noise_list', exist_ok=True)
    for ntype in NTYPES:
        if ntype=='wham_noise':
            for set0 in ['train', 'test']:
                noise_files_lst = f'data/noise_list/{set0}_noiselist_{ntype}.csv'
                if set0 == 'train':
                    lst_ns = add_noise.get_noise_files_new('wham_noise/tr')
                else:
                    lst_ns = add_noise.get_noise_files_new('wham_noise/cv')
                with open(noise_files_lst, 'w') as file: # pylint: disable=unspecified-encoding
                    for name in lst_ns:
                        name = re.sub(r'\\', '/', name)
                        file.write(f'{name}\n')
        elif ntype in {'FSD50K','ESC-50-MASTER'}:
            with open(f'wavs/noise/{ntype}/non_speech.csv', 'r') as file: # pylint: disable=unspecified-encoding
                lines = file.readlines()
            random.shuffle(lines)
            lst_ns={}
            start = int(len(lines) / 5)

            lst_ns['train'] = lines[start:]
            lst_ns['test'] = lines[:start]

            for set0 in ['train', 'test']:
                noise_files_lst = f'data/noise_list/{set0}_noiselist_{ntype}.csv'

                with open(noise_files_lst, 'w') as file: # pylint: disable=unspecified-encoding
                    for name in lst_ns[set0]:
                        name = re.sub(r'\\', '/', name)
                        file.write(f'{name}')
                    if ntype=='ESC-50-MASTER':
                        lst_must = add_noise.get_noise_files_new('others/must')
                        for name in lst_must:
                            name = re.sub(r'\\', '/', name)
                            file.write(f'{name}\n')
        elif ntype in {'musan'}:
            lst_music = add_noise.get_noise_files_new('musan/music')
            lst_noise = add_noise.get_noise_files_new('musan/noise')
            lines = lst_music + lst_noise
            random.shuffle(lines)
            lst_ns={}
            start = int(len(lines) / 5)
            lst_ns['test'] = lines[:start]
            lst_ns['train'] = lines[start:]
            for set0 in ['train', 'test']:
                noise_files_lst = f'data/noise_list/{set0}_noiselist_{ntype}.csv'

                with open(noise_files_lst, 'w') as file: # pylint: disable=unspecified-encoding
                    for name in lst_ns[set0]:
                        name = re.sub(r'\\', '/', name)
                        file.write(f'{name}\n')

        elif ntype in {'social_noise', 'traffic'}:
            lst_ns={}
            lst_ns['test'] = add_noise.get_noise_files_new(f'{ntype}/test')
            lst_ns['train'] = add_noise.get_noise_files_new(f'{ntype}/train')
            for set0 in ['train', 'test']:
                noise_files_lst = f'data/noise_list/{set0}_noiselist_{ntype}.csv'

                with open(noise_files_lst, 'w') as file: # pylint: disable=unspecified-encoding
                    for name in lst_ns[set0]:
                        name = re.sub(r'\\', '/', name)
                        file.write(f'{name}\n')
        else:
            ntype0 = re.sub(r'/', '_', ntype)
            noise_files_train = f'data/noise_list/train_noiselist_{ntype0}.csv'
            noise_files_test = f'data/noise_list/test_noiselist_{ntype0}.csv'
            lst_ns = add_noise.get_noise_files_new(ntype)
            random.shuffle(lst_ns)
            start = int(len(lst_ns) / 5)
            with open(noise_files_train, 'w') as file: # pylint: disable=unspecified-encoding
                for name in lst_ns[start:]:
                    name = re.sub(r'\\', '/', name)
                    file.write(f'{name}\n')

            with open(noise_files_test, 'w') as file:  # pylint: disable=unspecified-encoding
                for name in lst_ns[:start]:
                    name = re.sub(r'\\', '/', name)
                    file.write(f'{name}\n')

    target_files = { 'train': args.train_dataset_path,
                     'test' : args.test_dataset_path}
    tot_success_dict = {'train': [], 'test': []}
    for train_set in sets_categories:

        with open(target_files[train_set], 'r') as file: # pylint: disable=unspecified-encoding
            filepaths = file.readlines()
            random.shuffle(filepaths)
            if datasize_noise != -1:
                if train_set=='train':
                    filepaths = filepaths[:datasize_noise]
                else:
                    filepaths = filepaths[:int(datasize_noise / 5)]

        blk_size = int(np.floor(len(filepaths) / args.num_procs))
        sub_src = []
        for i in range(args.num_procs):
            idx0 = i * blk_size
            if i == args.num_procs - 1:
                sub_src += [filepaths[idx0:]]
            else:
                sub_src += [filepaths[idx0:blk_size+idx0]]

        for ntype in NTYPES:
            manager = multiprocessing.Manager()
            success_dict = manager.dict({i: [] for i in range(args.num_procs)})
            print(f'{train_set} set running: snr_dB = {SNR_DBS[0]}--{SNR_DBS[-1]}, ntype={ntype}')
            ntype0 = re.sub(r'/', '_', ntype)
            noise_files_train = f'data/noise_list/train_noiselist_{ntype0}.csv'
            noise_files_test = f'data/noise_list/test_noiselist_{ntype0}.csv'

            with open(noise_files_train) as file: # pylint: disable=unspecified-encoding
                lines = file.readlines()
            lines_tr = [line.strip() for line in lines]

            with open(noise_files_test) as file: # pylint: disable=unspecified-encoding
                lines = file.readlines()
            lines_te = [line.strip() for line in lines]

            noise_files = { 'train' : lines_tr,
                            'test'  : lines_te}
            processes = [
                FeatMultiProcsClass(
                        i, f"Thread-{i}",
                        sub_src[i],
                        train_set,
                        ntype,
                        noise_files,
                        SNR_DBS,
                        success_dict,
                        params_audio_def = params_audio,
                        num_procs = args.num_procs,
                        reverb_lst = lst_reverb[train_set],
                        reverb_prob = reverb_prob)
                            for i in range(args.num_procs)]

            start_time = time.time()

            if DEBUG:
                for proc in processes:
                    proc.run()
            else:
                for proc in processes:
                    proc.start()

                for proc in processes:
                    proc.join()
                print(f"\nTime elapse {time.time() - start_time} sec")

            if args.wandb_track:
                data = wandb.Artifact(
                    S3_BUCKET + "-tfrecords",
                    type="dataset",
                    description="tfrecords of speech command dataset")
                data.add_reference(f"s3://{S3_BUCKET}/{S3_PREFIX}", max_objects=31000)
                run.log_artifact(data)

            for lst in success_dict.values():
                tot_success_dict[train_set] += lst

    if not DEBUG:
        for train_set in sets_categories:
            with open(f"data/{train_set}_tfrecords_se.csv", 'w') as file: # pylint: disable=unspecified-encoding
                random.shuffle(tot_success_dict[train_set])
                for tfrecord in tot_success_dict[train_set]:
                    tfrecord = re.sub(r'\\', '/', tfrecord)
                    file.write(f'{tfrecord}\n')

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    argparser = argparse.ArgumentParser(
        description='Generate TFrecord formatted input data from a raw speech commands dataset')
    argparser.add_argument(
        '-d',
        '--download',
        type    = int,
        default = 0,
        help    = 'download training data')

    argparser.add_argument(
        '-rb',
        '--reverb_prob',
        type    = float,
        default = 0,
        help    = 'percentage of size for reverb dataset')

    argparser.add_argument(
        '-t',
        '--train_dataset_path',
        default = 'data/train_se.csv',
        help    = 'path to train data file')

    argparser.add_argument(
        '--test_dataset_path',
        default = 'data/test_se.csv',
        help    = 'path to test data file')

    argparser.add_argument(
        '-s',
        '--datasize_noise',
        type    = int,
        default = 30000,
        help='How many speech samples per noise')

    argparser.add_argument(
        '-n',
        '--num_procs',
        type    = int,
        default = 8,
        help='How many processor cores to use for execution')

    argparser.add_argument(
        '-w',
        '--wandb_track',
        default = False,
        help    = 'Enable tracking of this run in Weights&Biases')

    argparser.add_argument(
        '--wandb_project',
        type    = str,
        default = 'se',
        help='Weights&Biases project name')

    argparser.add_argument(
        '--wandb_entity',
        type    = str,
        default = 'ambiq',
        help    = 'Weights&Biases entity name')

    main(argparser.parse_args())
