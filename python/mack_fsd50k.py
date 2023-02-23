from pathlib import Path
import pandas as pd
import logging
from random import sample
import fire
import pdb

debug = False 


def main():
    pass


def generate_fsd_splits(fsd_base):
    dev_labels = Path(fsd_base) / 'FSD50K.ground_truth' / 'dev.csv'
    eval_labels = Path(fsd_base) / 'FSD50K.ground_truth' / 'eval.csv'
    wav_dir = Path(fsd_base) / 'FSD50K.dev_audio'
    dev_df = pd.read_csv(str(dev_labels))
    eval_df = pd.read_csv(str(dev_labels))

    noises = ['Computer_keyboard', 'Clapping',
              'Crumpling_and_crinkling',
              'Dishes_and_pots_and_pans',
              'Mechanical_fan', 'Packing_tape_and_duct_tape']
    rejects = ['Speech', 'Human_voice']

    def noise_filter(x, noise):
        if noise not in x:
            return False
        # reject any noise samples that have a class of interest in them
        return set(x) & set(rejects) == set()
    test_samples = []
    train_samples = []
    for noise in noises:
        noise_samples = []
        labels = dev_df['labels']
        labels = labels.apply(lambda x: x.split(','))
        label_allow_list = labels.apply(lambda x: noise_filter(x, noise))

        class_data = dev_df[label_allow_list]

        # why randomly split if we are already filtering...
        train_data = class_data[class_data['split'] == 'train']
        val_data = class_data[class_data['split'] == 'val']
        logging.info(
            f"found {len(class_data.index)} items for {noise}, split of {len(train_data.index)} / {len(val_data.index)}")

        train_samples += (train_data['fname'].map(str) + ".wav").tolist()
        test_samples += (val_data['fname'].map(str) + ".wav").tolist()
        noise_samples = (class_data['fname'].map(
            str) + ",    " + class_data['labels']).tolist()
        if debug:
            with open(f"{noise}_samples.csv", 'w') as file:
                file.write("\n".join(noise_samples))

    train_samples = [str(wav_dir / x) for x in train_samples]
    test_samples = [str(wav_dir / x) for x in test_samples]
    with open("train_noiselist_FSD50K.csv", 'w') as file:
        file.write("\n".join(train_samples))

    with open("test_noiselist_FSD50K.csv", 'w') as file:
        file.write("\n".join(test_samples))


if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s %(filename)s:%(lineno)s  %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fire.Fire(generate_fsd_splits)
