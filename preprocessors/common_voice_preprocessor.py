import os
import numpy as np
import pandas as pd
from audio import mel_spectrogram
from tqdm import tqdm


def preprocess(in_path, out_path, stft):
    entries = []
    data = pd.read_csv(os.path.join(in_path, 'validated.tsv'), delimiter='\t', escapechar='\\', quotechar="'")
    for _, row in tqdm(list(data.iterrows())):
        speaker, audio, text = row['client_id'], row['path'], row['sentence']
        mel = mel_spectrogram(os.path.join(in_path, 'clips', audio), stft)
        mel_len = mel.size(1)
        mel_file = f'{os.path.splitext(audio)[0]}.npy'
        np.save(os.path.join(out_path, mel_file), mel, allow_pickle=False)
        entries.append((text, audio, mel_file, mel_len, speaker))

    entries = pd.DataFrame(entries, columns=['text', 'audio', 'mel', 'mel_len', 'speaker'])
    entries.to_csv(os.path.join(out_path, 'data.csv'), index=False)
