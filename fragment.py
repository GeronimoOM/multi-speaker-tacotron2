import argparse
import pandas as pd
from hparams import create_hparams
import os


def fragment(data_path, out_path, hparams):
    data = pd.read_csv(data_path)
    F = hparams.n_fragment_mel_windows
    data['n_samples'] = data['mel_len'] // F
    entries = []
    for _, row in data.iterrows():
        entries += [(row['mel'], str(row['speaker']), int(i*F), int((i+1)*F)) for i in range(row['n_samples'])]

    data_file_name = os.path.splitext(os.path.basename(data_path))[0]
    fragments = pd.DataFrame(entries, columns=['mel', 'speaker', 'from', 'to'])
    fragments.to_csv(os.path.join(out_path, f'{data_file_name}_fragments.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str)
    parser.add_argument('-o', '--output_directory', type=str)

    args = parser.parse_args()

    hparams = create_hparams('speaker_encoder')

    fragment(args.data_path, args.output_directory, hparams)


