import random
import os
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from text import text_to_sequence


class TextMelDataset(torch.utils.data.Dataset):
    def __init__(self, data_files, device, hparams):
        self.data = []
        for data_file in data_files if isinstance(data_files, list) else [data_files]:
            data = pd.read_csv(data_file)
            data['mel'] = data['mel'].apply(lambda mel: os.path.join(os.path.split(data_file)[0], mel))
            self.data.append(data)
        self.data = pd.concat(self.data)
        self.device = device
        self.text_cleaners = hparams.text_cleaners
        random.seed(hparams.seed)

    def get_mel(self, audio_path):
        mel = torch.from_numpy(np.load(audio_path))
        return mel.transpose(0, 1).to(device=self.device)

    def get_text(self, text):
        text_norm = torch.tensor(text_to_sequence(text, self.text_cleaners), dtype=torch.long, device=self.device)
        return text_norm

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = self.get_text(row['text'])
        mel = self.get_mel(row['mel'])
        return text, mel

    def __len__(self):
        return len(self.data)


class TextMelCollate:
    def __call__(self, batch):
        device = batch[0][0].device
        B = len(batch)
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.tensor([len(x[0]) for x in batch], dtype=torch.int), dim=0, descending=True)
        T = input_lengths[0]

        text_padded = torch.zeros(B, T, dtype=torch.long, device=device)
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        S = max([x[1].size(0) for x in batch])
        M = batch[0][1].size(1)

        mel_padded = torch.zeros(B, S, M, device=device)
        gate_padded = torch.zeros(B, S, device=device)
        output_lengths = torch.empty(B, dtype=torch.int, device=device)
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            s = mel.size(0)
            mel_padded[i, :s] = mel
            gate_padded[i, s-1:] = 1
            output_lengths[i] = s

        return (text_padded, input_lengths, mel_padded, output_lengths), (mel_padded, gate_padded)


class MelFragmentDataset(torch.utils.data.IterableDataset):

    def __init__(self, fragments_files, batch_size_speakers, batch_size_speaker_samples, device, hparams):
        self.fragments = []
        for fragment_file in fragments_files if isinstance(fragments_files, list) else [fragments_files]:
            fragments = pd.read_csv(fragment_file)
            fragments['mel'] = fragments['mel'].apply(lambda mel: os.path.join(os.path.split(fragment_file)[0], mel))
            self.fragments.append(fragments)
        self.fragments = pd.concat(self.fragments)

        self.n_mel_channels = hparams.n_mel_channels
        self.n_fragment_mel_windows = hparams.n_fragment_mel_windows
        self.speaker_fragments = {}
        self.batch_size_speakers = batch_size_speakers
        self.batch_size_speaker_samples = batch_size_speaker_samples

        for _, row in self.fragments.iterrows():
            self.speaker_fragments.setdefault(row['speaker'], []).append((row['mel'], row['from'], row['to']))

        self.speaker_fragments = {s: fs for s, fs in self.speaker_fragments.items()
                                  if len(fs) >= self.batch_size_speaker_samples}

        self.speaker_count = len(self.speaker_fragments)

        speaker_fragment_counts = [len(fs) for fs in self.speaker_fragments.values()]
        self.fragment_count = sum(speaker_fragment_counts)
        self.speaker_p = np.array(speaker_fragment_counts) / self.fragment_count

        self.batch_size = self.batch_size_speakers * self.batch_size_speaker_samples
        self.batch_count = self.fragment_count // self.batch_size

        self.device = device
        random.seed(hparams.seed)

    def __len__(self):
        return self.batch_count

    def __iter__(self):
        return MelFragmentIter(self)

    def get_fragment(self, mel_path, fr, to):
        return torch.from_numpy(np.load(mel_path)[:, fr:to])


class MelFragmentIter:

    def __init__(self, dataset: MelFragmentDataset):
        self.dataset = dataset
        self.last_batch = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.last_batch == self.dataset.batch_count:
            raise StopIteration

        self.last_batch += 1

        # B, M, T
        mels = torch.empty(self.dataset.batch_size, self.dataset.n_fragment_mel_windows, self.dataset.n_mel_channels,
                           device=self.dataset.device)
        speakers = list(self.dataset.speaker_fragments.keys())

        if self.dataset.speaker_count > self.dataset.batch_size_speakers:
            speakers = np.random.choice(speakers, self.dataset.batch_size_speakers,
                                        replace=False, p=self.dataset.speaker_p)

        i = 0
        for s in speakers:
            fragments = self.dataset.speaker_fragments[s]
            fragments_indices = range(len(fragments))
            if len(fragments) > self.dataset.batch_size_speaker_samples:
                fragments_indices = np.random.choice(fragments_indices, self.dataset.batch_size_speaker_samples,
                                                     replace=False)
            for f in fragments_indices:
                mel, fr, to = fragments[f]
                mels[i] = self.dataset.get_fragment(mel, fr, to).transpose(0, 1)
                i += 1

        return mels, speakers



