import numpy as np
import pandas as pd
import os
from audio import mel_spectrogram
from tqdm import tqdm


def preprocess(in_path, out_path, stft):
    speakers = ["Meelis_Kompus", "Tarmo_Maiberg", "Birgit_Itse", "Vallo_Kelmsaar", "Indrek_Kiisler",
                "Tõnu_Karjatse", "Kai_Vare", "Katarina", "Kristo", "Robert", "Stella"]

    for speaker_id, speaker in tqdm(enumerate(speakers)):
        if os.path.exists(os.path.join(out_path, f'{speaker}_data.csv')):
            continue

        speaker_entries = []
        speaker_data = pd.read_csv(os.path.join(in_path, speaker, 'sentences_filtered.csv'),
                                   escapechar='\\', quotechar="'", header=None)
        for audio, text in tqdm(list(speaker_data.itertuples(index=False))):
            mel = mel_spectrogram(os.path.join(in_path, speaker, audio), stft)
            mel_len = mel.size(1)
            mel_file = f'{speaker}_{os.path.splitext(audio)[0]}.npy'
            np.save(os.path.join(out_path, mel_file), mel, allow_pickle=False)
            speaker_entries.append((text, os.path.join(speaker, audio), mel_file, mel_len, speaker_id))
        pd.DataFrame(speaker_entries, columns=['text', 'audio', 'mel', 'mel_len', 'speaker'])\
            .to_csv(os.path.join(out_path, f'{speaker}_data.csv'), index=False)

    entries = pd.concat([pd.read_csv(os.path.join(out_path, f'{speaker}_data.csv')) for speaker in speakers])
    entries.to_csv(os.path.join(out_path, 'data.csv'), index=False)
