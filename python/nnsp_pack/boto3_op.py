"""
some boto3 functions
"""
import os
import re
import boto3
def s3_download(s3_bucket, audio_lists):
    """
    Downloading s2i data from aws s3
    """
    s3_resource = boto3.resource('s3')
    bucket = s3_resource.Bucket(s3_bucket)

    # target-audio & noise files downloading
    for audio_list in audio_lists:

        if audio_list in [  'data/noise_list.csv',
                            'data/train_garb.csv',
                            'data/test_garb.csv']:
            with open(audio_list, 'r') as file: # pylint: disable=W1514
                lines = file.readlines()
        else:
            with open(audio_list, 'r') as file: # pylint: disable=W1514
                lines = file.readlines()[1:]

        fnames = [re.sub(r'\\', '/', line.strip().split(',')[0]) for line in lines]

        for fname in fnames:
            fname_src = re.sub(r'(\./wavs/|wavs/)', '', fname)
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            if os.path.exists(fname):
                pass
            else:
                try:
                    bucket.download_file(fname_src, fname)
                except: # pylint: disable=W0702
                    print(f'downloading {fname} failed')
                else:
                    print(f'downloading {fname} succeeded')
