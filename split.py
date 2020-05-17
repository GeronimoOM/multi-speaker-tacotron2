import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from hparams import create_hparams


def train_test_val_split(data, test_size, val_size, hparams):
    train, test = train_test_split(data, test_size=test_size, random_state=hparams.seed, stratify=data['speaker'])
    train, val = train_test_split(data, test_size=len(data['speaker'].unique())*val_size,
                                  random_state=hparams.seed, stratify=data['speaker'])
    return train, test, val


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str)
    parser.add_argument('-m', '--model', type=str, default='tacotron')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')

    args = parser.parse_args()

    model = args.model
    hparams = create_hparams(args.model, args.hparams)

    data = pd.read_csv(args.data, names=['text', 'mel', 'mel_len', 'speaker'], skiprows=1)

    train, test, val = train_test_val_split(data, 0.1, 10, hparams)

    out_path = os.path.dirname(os.path.realpath(args.data))
    pd.DataFrame(train).to_csv(os.path.join(out_path, 'train.csv'), index=False)
    pd.DataFrame(test).to_csv(os.path.join(out_path, 'test.csv'), index=False)
    pd.DataFrame(val).to_csv(os.path.join(out_path, 'val.csv'), index=False)
