import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from hparams import create_hparams


def split(data_path, out_path, test_size, hparams):
    data = pd.read_csv(data_path)
    train, test = train_test_split(data, test_size=test_size, random_state=hparams.seed)
    train, val = train_test_split(train, test_size=hparams.batch_size*3, random_state=hparams.seed)

    file_name = os.path.splitext(os.path.basename(data_path))[0]
    train.to_csv(os.path.join(out_path, f'{file_name}_train.csv'), index=False)
    test.to_csv(os.path.join(out_path, f'{file_name}_test.csv'), index=False)
    val.to_csv(os.path.join(out_path, f'{file_name}_val.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str, default=None)
    parser.add_argument('-o', '--output_directory', type=str)
    parser.add_argument('-p', '--test_size', type=float, default=0.2)
    parser.add_argument('--hparams', type=str,
                        default=None, help='comma separated name=value pairs')

    args = parser.parse_args()

    hparams = create_hparams('tacotron', args.hparams)
    split(args.data_path, args.output_directory, args.test_size, hparams)


