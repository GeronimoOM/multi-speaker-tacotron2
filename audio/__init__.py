import torch
import librosa
from audio.stft import TacotronSTFT


def init_stft(hparams):
    return TacotronSTFT(
        hparams.filter_length, hparams.hop_length, hparams.win_length,
        hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
        hparams.mel_fmax)


def mel_spectrogram(audio_path, stft):
    audio, sampling_rate = librosa.core.load(audio_path, sr=stft.sampling_rate)
    audio = torch.from_numpy(audio).unsqueeze(0)
    mel = stft.mel_spectrogram(audio)
    mel = torch.squeeze(mel, 0)
    return mel
