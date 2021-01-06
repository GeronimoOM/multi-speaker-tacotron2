import argparse
import pandas as pd
import os
import numpy as np
import torch
import librosa as lrs

from hparams import create_hparams
from model import Tacotron2
from text import text_to_sequence as text2seq

from waveglow.denoiser import Denoiser

checkpoints_root = ''
seen_mels_root = ''
unseen_mels_root = ''


def infer(inputs_file, models_file, out_path):
    inputs = pd.read_csv(inputs_file)
    models = pd.read_csv(models_file)

    waveglow, denoiser = init_vocoder()

    output_columns = ['id', 'model', 'audio', 'speaker', 'seen', 'text',
                      'speaker_audio', 'speaker_mel', 'speaker_mel_len']

    inputs_filename = os.path.splitext(os.path.split(inputs_file)[1])[0]
    output_file = f'output_{inputs_filename}.csv'
    output = []
    n = 0
    checkpoint_every = 30
    for i, (_, model_row) in enumerate(models.iterrows()):
        model = init_model(
            os.path.join(checkpoints_root, model_row['tacotron']),
            os.path.join(checkpoints_root, model_row['speaker_encoder']))

        input_rows = inputs[inputs['model'] == i]
        for _, input_row in input_rows.iterrows():
            sequence = torch.tensor(text_to_sequence(input_row['text']), dtype=torch.long)
            mel_input = get_mel(os.path.join(seen_mels_root if input_row['seen'] else unseen_mels_root, input_row['mel']))

            audio = infer_one(sequence, mel_input, model, waveglow, denoiser)

            id = input_row['id']
            out_file = f'{id}.wav'
            lrs.output.write_wav(os.path.join(out_path, out_file), audio.data.numpy()[0], 22050)
            output.append((
                id, model_row['model'], out_file, input_row['speaker'], input_row['seen'], input_row['text'],
                input_row['audio'], input_row['mel'], input_row['mel_len']
            ))

            n += 1
            if n % checkpoint_every == 0:
                pd.DataFrame(output, columns=output_columns).to_csv(
                    os.path.join(out_path, output_file), index=False)

    pd.DataFrame(output, columns=output_columns).to_csv(
        os.path.join(out_path, output_file), index=False)


def text_to_sequence(text):
    return torch.tensor(text2seq(text, ['english_cleaners']), dtype=torch.long)


def get_mel(audio_path):
    mel = torch.from_numpy(np.load(audio_path))
    return mel


def init_model(tacotron_cp, speaker_encode_cp):
    hparams = create_hparams('tacotron',
                             f'speaker_encoder={speaker_encode_cp}'
                             if speaker_encode_cp is not None else None)

    model = Tacotron2(hparams)
    model.load_state_dict(torch.load(tacotron_cp)['state_dict'])
    model.eval()

    return model


def init_vocoder():
    waveglow_path = '../waveglow_256channels_universal_v5.pt'
    waveglow = torch.load(waveglow_path)['model']
    for k in waveglow.convinv:
        k.float()
    denoiser = Denoiser(waveglow)
    return waveglow, denoiser


def infer_one(sequence, mel_input, model, waveglow, denoiser, denoise=True):
    with torch.no_grad():
        mel_outputs, mel_outputs_postnet, _, alignments = \
            model.inference((sequence.unsqueeze(0), mel_input.T.unsqueeze(0)))
        audio = waveglow.infer(mel_outputs_postnet.transpose(1, 2), sigma=0.666)[0]
        if denoise:
            audio = denoiser(audio.unsqueeze(0), strength=0.01)[:, 0]
    return audio


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--input_file', type=str)
    parser.add_argument('-m', '--models_file', type=str)
    parser.add_argument('-o', '--output_directory', type=str)

    args = parser.parse_args()
    os.makedirs(args.output_directory)
    infer(args.input_file, args.models_file, args.output_directory)





