import numpy as np
import pandas as pd
import os
from audio import init_stft, mel_spectrogram
from tqdm import tqdm


def preprocess(in_path, out_path, hparams):
    speakers = ["Meelis_Kompus", "Tarmo_Maiberg", "Birgit_Itse", "Vallo_Kelmsaar", "Indrek_Kiisler",
                "Tõnu_Karjatse", "Kai_Vare", "Katarina", "Kristo", "Robert", "Stella"]

    stft = init_stft(hparams)
    for speaker_id, speaker in tqdm(enumerate(speakers)):
        if os.path.exists(os.path.join(out_path, f'{speaker}_data.csv')):
            continue
        speaker_entries = []
        speaker_data = pd.read_csv(os.path.join(in_path, speaker, 'sentences_filtered.csv'),
                                   escapechar='\\', quotechar="'", header=None)
        for audio_file, text in tqdm(list(speaker_data.itertuples(index=False))):
            audio_path = os.path.join(in_path, speaker, audio_file)
            mel = mel_spectrogram(audio_path, stft)
            mel_windows = mel.size(1)
            mel_path = os.path.join(out_path, f'{speaker}_{os.path.splitext(audio_file)[0]}.npy')
            np.save(mel_path, mel, allow_pickle=False)
            speaker_entries.append((text, mel_path, mel_windows, speaker_id))
        pd.DataFrame(speaker_entries).to_csv(os.path.join(out_path, f'{speaker}_data.csv'), index=False)

    entries = [pd.read_csv(os.path.join(out_path, f'{speaker}_data.csv')) for speaker in speakers]
    pd.concat(entries).to_csv(os.path.join(out_path, 'data.csv'), index=False)
