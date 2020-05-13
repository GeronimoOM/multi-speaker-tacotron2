import numpy as np
import pandas as pd
import os
import csv
from audio import init_stft, mel_spectrogram
from tqdm import tqdm


def preprocess(in_path, out_path, hparams):
    #speakers = ["Meelis_Kompus", "Tarmo_Maiberg", "Birgit_Itse", "Vallo_Kelmsaar", "Indrek_Kiisler",
    #            "Tõnu_Karjatse", "Kai_Vare", "Katarina", "Kristo", "Robert", "Stella"]
    speakers = ["Stella"]

    entries = []
    stft = init_stft(hparams)
    for speaker_id, speaker in tqdm(enumerate(speakers)):
        with open(os.path.join(in_path, speaker, 'sentences_filtered.csv'), encoding='utf-8') as f:
            for audio_file, text in tqdm(csv.reader(f, delimiter=',', escapechar='\\', quotechar="'")):
                audio_path = os.path.join(in_path, speaker, audio_file)
                mel = mel_spectrogram(audio_path, stft)
                mel_windows = mel.size(1)
                mel_path = os.path.join(out_path, f'{os.path.splitext(audio_file)[0]}.npy')
                np.save(mel_path, mel, allow_pickle=False)
                entries.append((text, mel_path, mel_windows, speaker_id))

    audio_paths_and_text = pd.DataFrame(entries)
    audio_paths_and_text.to_csv(os.path.join(out_path, 'data.csv'), index=False)
