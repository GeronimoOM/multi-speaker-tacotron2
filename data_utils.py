import random
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from text import text_to_sequence
from audio import init_stft, mel_spectrogram


class TextMelDataset(torch.utils.data.Dataset):
    def __init__(self, audio_paths_and_text, device, hparams):
        self.audio_paths_and_text = pd.read_csv(audio_paths_and_text).to_numpy()
        self.device = device
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.text_cleaners = hparams.text_cleaners
        self.stft = init_stft(hparams)
        random.seed(hparams.seed)
        random.shuffle(self.audio_paths_and_text)

    def get_mel(self, audio_path):
        if not self.load_mel_from_disk:
            mel = mel_spectrogram(audio_path, self.stft)
        else:
            mel = torch.from_numpy(np.load(audio_path))

        return mel.transpose(0, 1).to(device=self.device)

    def get_text(self, text):
        text_norm = torch.tensor(text_to_sequence(text, self.text_cleaners), dtype=torch.long, device=self.device)
        return text_norm

    def __getitem__(self, idx):
        text, audio_path, _, _ = self.audio_paths_and_text[idx]
        text = self.get_text(text)
        mel = self.get_mel(audio_path)
        return text, mel

    def __len__(self):
        return len(self.audio_paths_and_text)


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

    def __init__(self, fragment_paths, device, hparams):
        fragment_paths = pd.read_csv(fragment_paths)

        self.n_mel_channels = hparams.n_mel_channels
        self.n_fragment_mel_windows = hparams.n_fragment_mel_windows
        self.speaker_fragments = {}
        self.batch_size_speakers = hparams.batch_size_speakers
        self.batch_size_speaker_samples = hparams.batch_size_speaker_samples

        for mel_path, speaker, fr, to in fragment_paths.itertuples(index=False):
            self.speaker_fragments.setdefault(speaker, []).append((mel_path, fr, to))

        self.speaker_fragments = [fs for fs in self.speaker_fragments.values()
                                  if len(fs) >= self.batch_size_speaker_samples]

        self.speaker_count = len(self.speaker_fragments)

        speaker_fragment_counts = [len(fs) for fs in self.speaker_fragments]
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

        speaker_indices = np.random.choice(range(self.dataset.speaker_count), self.dataset.batch_size_speakers,
                                           replace=False, p=self.dataset.speaker_p)

        # B, M, T
        mels = torch.empty(self.dataset.batch_size, self.dataset.n_fragment_mel_windows, self.dataset.n_mel_channels,
                           device=self.dataset.device)
        speakers = torch.empty(self.dataset.batch_size, dtype=torch.int)

        i = 0
        for s in speaker_indices:
            fragments = self.dataset.speaker_fragments[s]
            fragments_indices = np.random.choice(range(len(fragments)), self.dataset.batch_size_speaker_samples,
                                                 replace=False)
            for f in fragments_indices:
                mel, fr, to = fragments[f]
                mels[i] = self.dataset.get_fragment(mel, fr, to).transpose(0, 1)
                speakers[i] = s
                i += 1

        return mels, speakers



