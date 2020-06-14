import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from hparams import create_hparams


def split_fragments(fragments_path, out_path, test_size, hparams):
    data = pd.read_csv(fragments_path)
    train, test = train_test_split(data, test_size=test_size, stratify=data['speaker'], random_state=hparams.seed)

    N = hparams.batch_size_speakers
    M = hparams.batch_size_speaker_samples
    val_speakers = np.random.choice(data['speaker'].unique(), size=N, replace=False)
    val = pd.concat([train.loc[train['speaker'] == s].sample(M, replace=False, random_state=hparams.seed)
                     for s in val_speakers])
    train = train.drop(val.index)

    fragments_file_name = os.path.splitext(os.path.basename(fragments_path))[0]
    train.to_csv(os.path.join(out_path, f'{fragments_file_name}_train.csv'), index=False)
    test.to_csv(os.path.join(out_path, f'{fragments_file_name}_test.csv'), index=False)
    val.to_csv(os.path.join(out_path, f'{fragments_file_name}_val.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str, default=None)
    parser.add_argument('-o', '--output_directory', type=str)
    parser.add_argument('-p', '--test_size', type=float, default=0.3)

    args = parser.parse_args()

    hparams = create_hparams('speaker_encoder')
    split_fragments(args.data_path, args.output_directory, args.test_size, hparams)


