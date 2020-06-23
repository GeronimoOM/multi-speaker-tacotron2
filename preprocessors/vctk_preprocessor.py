import numpy as np
import pandas as pd
import os
from audio import mel_spectrogram
from tqdm import tqdm


def preprocess(in_path, out_path, stft):
    speakers = []
    for dirpath, dirnames, filenames in tqdm(os.walk(os.path.join(in_path, 'txt'))):
        if not len(filenames):
            continue

        speaker_id = os.path.basename(dirpath)
        speakers.append(speaker_id)
        if os.path.exists(os.path.join(out_path, f'{speaker_id}_data.csv')):
            continue

        speaker_entries = []
        for filename in tqdm(filenames):
            if not filename.endswith('.txt'):
                continue
            with open(os.path.join(dirpath, filename), 'r', encoding='ISO-8859-1') as f:
                text = f.read()
            entryname = os.path.splitext(filename)[0]

            audio_path = os.path.join(in_path, 'wav48', speaker_id, f'{entryname}.wav')
            mel = mel_spectrogram(audio_path, stft)
            mel_len = mel.size(1)
            mel_path = os.path.join(out_path, f'{entryname}.npy')
            np.save(mel_path, mel, allow_pickle=False)
            speaker_entries.append((text, mel_path, mel_len, speaker_id))

        pd.DataFrame(speaker_entries, columns=['text', 'mel', 'mel_len', 'speaker'])\
            .to_csv(os.path.join(out_path, f'{speaker_id}_data.csv'), index=False)

    entries = pd.concat([pd.read_csv(os.path.join(out_path, f'{speaker}_data.csv')) for speaker in speakers])
    entries.to_csv(os.path.join(out_path, 'data.csv'), index=False)
