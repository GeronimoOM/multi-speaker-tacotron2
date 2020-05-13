import numpy as np
import pandas as pd
import os
import csv
from audio import init_stft, mel_spectrogram
from tqdm import tqdm


def preprocess(in_path, out_path, hparams):
    entries = []
    stft = init_stft(hparams)
    with open(os.path.join(in_path, 'validated.tsv'), encoding='utf-8') as f:
        row_iter = csv.reader(f, delimiter='\t', escapechar='\\', quotechar="'")
        next(row_iter)  # skip header
        for row in tqdm(row_iter):
            speaker_id, audio_file, text = row[0], row[1], row[2]
            audio_path = os.path.join(in_path, 'clips', audio_file)
            mel = mel_spectrogram(audio_path, stft)
            mel_windows = mel.size(1)
            mel_path = os.path.join(out_path, f'{os.path.splitext(audio_file)[0]}.npy')
            np.save(mel_path, mel, allow_pickle=False)
            entries.append((text, mel_path, mel_windows, speaker_id))

    audio_paths_and_text = pd.DataFrame(entries)
    audio_paths_and_text.to_csv(os.path.join(out_path, 'data.csv'), index=False)
