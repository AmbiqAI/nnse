"""
Make training and testing data list
"""
import os
import re
import random

fnames = []
for root, folders, files in os.walk('./wavs/vad'):
    for f in files:
        if re.search(r'(\.wav$|\.flac$)', f):
            fnames += [os.path.join(root, f)]

random.shuffle(fnames)

fnames_te = fnames[:int(len(fnames)/5)]
with open('data/test_files_vad.csv', 'w') as file:
    for fname in fnames_te:
        file.write(f"{fname}\n")

fnames_tr = fnames[int(len(fnames)/5):]
with open('data/train_files_vad.csv', 'w') as file:
    for fname in fnames_tr:
        file.write(f"{fname}\n")
