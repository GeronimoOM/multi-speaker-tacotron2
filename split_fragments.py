import os
import argparse
import numpy as np
import pandas as pd
from hparams import create_hparams


def split_fragments(fragments_path, out_path, hparams):
    data = pd.read_csv(fragments_path)

    N = hparams.batch_size_speakers
    M = hparams.batch_size_speaker_samples
    speaker_fragments = data['speaker'].value_counts()
    val_speakers = np.random.choice(speaker_fragments[speaker_fragments >= M].index.unique(), size=N, replace=False)
    val = pd.concat([data.loc[data['speaker'] == s].sample(M, replace=False, random_state=hparams.seed)
                     for s in val_speakers])
    train = data.drop(val.index)

    fragments_file_name = os.path.splitext(os.path.basename(fragments_path))[0]
    train.to_csv(os.path.join(out_path, f'{fragments_file_name}_train.csv'), index=False)
    val.to_csv(os.path.join(out_path, f'{fragments_file_name}_val.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str, default=None)
    parser.add_argument('-o', '--output_directory', type=str)
    parser.add_argument('--hparams', type=str,
                        default=None, help='comma separated name=value pairs')

    args = parser.parse_args()

    hparams = create_hparams('speaker_encoder', args.hparams)
    split_fragments(args.data_path, args.output_directory, hparams)


