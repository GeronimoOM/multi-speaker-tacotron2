import os
import numpy as np
import pandas as pd
from audio import mel_spectrogram
from tqdm import tqdm


def preprocess(in_path, out_path, stft):
    entries = []
    data = pd.read_csv(os.path.join(in_path, 'validated.tsv'), delimiter='\t', escapechar='\\', quotechar="'")
    for _, row in tqdm(list(data.iterrows())):
        speaker, audio_file, text = row['client_id'], row['path'], row['sentence']
        audio_path = os.path.join(in_path, 'clips', audio_file)
        mel = mel_spectrogram(audio_path, stft)
        mel_len = mel.size(1)
        mel_path = os.path.join(out_path, f'{os.path.splitext(audio_file)[0]}.npy')
        np.save(mel_path, mel, allow_pickle=False)
        entries.append((text, mel_path, mel_len, speaker))

    entries = pd.DataFrame(entries, columns=['text', 'mel', 'mel_len', 'speaker'])
    entries.to_csv(os.path.join(out_path, 'data.csv'), index=False)
