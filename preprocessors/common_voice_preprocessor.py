import os
import numpy as np
import pandas as pd
from audio import mel_spectrogram
from tqdm import tqdm


def preprocess(in_path, out_path, stft):
    entries = []
    data = pd.read_csv(os.path.join(in_path, 'validated.tsv'), delimiter='\t', escapechar='\\', quotechar="'")
    for speaker_id, audio_file, text in tqdm(list(data.itertuples(index=False))):
        audio_path = os.path.join(in_path, 'clips', audio_file)
        mel = mel_spectrogram(audio_path, stft)
        mel_windows = mel.size(1)
        mel_path = os.path.join(out_path, f'{os.path.splitext(audio_file)[0]}.npy')
        np.save(mel_path, mel, allow_pickle=False)
        entries.append((text, mel_path, mel_windows, speaker_id))

    entries = pd.DataFrame(entries, columns=['text', 'mel', 'mel_len', 'speaker'])
    entries.to_csv(os.path.join(out_path, 'data.csv'), index=False)
