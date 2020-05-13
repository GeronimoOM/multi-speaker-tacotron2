import random
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from text import text_to_sequence
from audio import init_stft, mel_spectrogram


class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, audio_paths_and_text, hparams):
        self.audio_paths_and_text = pd.read_csv(audio_paths_and_text).to_numpy()
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
            assert mel.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    mel.size(0), self.stft.n_mel_channels))

        return mel

    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
        return text_norm

    def __getitem__(self, idx):
        text, audio_path, _, speaker = self.audio_paths_and_text[idx]
        text = self.get_text(text)
        mel = self.get_mel(audio_path)
        return text, mel

    def __len__(self):
        return len(self.audio_paths_and_text)


class TextMelCollate:
    """ Zero-pads model inputs and targets based on number of frames per step
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collates training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)

        return (text_padded, input_lengths, mel_padded, output_lengths), \
               (mel_padded, gate_padded)


class MelFragmentLoader(torch.utils.data.IterableDataset):

    def __init__(self, mel_paths_and_text, hparams):
        mel_paths_and_text = pd.read_csv(mel_paths_and_text).to_numpy()

        self.n_mel_channels = hparams.n_mel_channels
        self.n_fragment_mel_windows = hparams.n_fragment_mel_windows
        self.speaker_fragments = {}
        self.batch_size_speakers = hparams.batch_size_speakers
        self.batch_size_speaker_samples = hparams.batch_size_speaker_samples

        for _, mel_path, mel_windows, speaker in mel_paths_and_text:
            n_mel_fragments = mel_windows // self.n_fragment_mel_windows
            for n in range(n_mel_fragments):
                self.speaker_fragments.setdefault(speaker, []).append((mel_path, n))

        self.speaker_fragments = [fs for fs in self.speaker_fragments.values()
                                  if len(fs) >= self.batch_size_speaker_samples]

        self.speaker_count = len(self.speaker_fragments)

        speaker_fragment_counts = [len(fs) for fs in self.speaker_fragments]
        self.fragment_count = sum(speaker_fragment_counts)
        self.speaker_p = np.array(speaker_fragment_counts) / self.fragment_count

        random.seed(hparams.seed)

    def __iter__(self):
        return MelFragmentIter(self)

    def get_fragment(self, mel_path, n):
        return torch.from_numpy(np.load(mel_path)[:, n*self.n_fragment_mel_windows:(n+1)*self.n_fragment_mel_windows])


class MelFragmentIter:

    def __init__(self, dataset: MelFragmentLoader):
        self.dataset = dataset

        self.batch_size = self.dataset.batch_size_speakers * self.dataset.batch_size_speaker_samples
        self.batch_count = self.dataset.fragment_count // self.batch_size * 5
        self.last_batch = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.last_batch == self.batch_count:
            raise StopIteration

        self.batch_count += 1

        speakers = np.random.choice(range(self.dataset.speaker_count), self.dataset.batch_size_speakers,
                                    replace=False, p=self.dataset.speaker_p)

        # B, M, T
        mels = torch.empty(self.batch_size, self.dataset.n_mel_channels, self.dataset.n_fragment_mel_windows)

        i = 0
        for s in speakers:
            fragments = self.dataset.speaker_fragments[s]
            fragments_indices = np.random.choice(range(len(fragments)), self.dataset.batch_size_speaker_samples,
                                                 replace=False)
            for f in fragments_indices:
                mel, n = fragments[f]
                mels[i] = self.dataset.get_fragment(mel, n)
                i += 1

        return mels, torch.from_numpy(speakers)



