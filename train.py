import os
import time
import argparse
from datetime import datetime
from itertools import chain

import torch
from torch.utils.data import DataLoader

from model import Tacotron2, SpeakerEncoder
from data_utils import TextMelDataset, TextMelCollate, MelFragmentDataset
from loss import Tacotron2Loss, SpeakerEncoderLoss
from logger import Tacotron2Logger, SpeakerEncoderLogger
from train_utils import load_checkpoint, save_checkpoint, warm_start_model
from hparams import create_hparams


def prepare_tacotron(device, output_directory, hparams):
    collate_fn = TextMelCollate()

    train_loader = DataLoader(TextMelDataset(hparams.data_train, device, hparams),
                              shuffle=True,
                              batch_size=hparams.batch_size,
                              drop_last=True, collate_fn=collate_fn)

    val_loaders = {
        'seen': DataLoader(TextMelDataset(hparams.data_val_seen, device, hparams),
                           shuffle=False,
                           batch_size=hparams.val_seen_size,
                           collate_fn=collate_fn),
        'unseen': DataLoader(TextMelDataset(hparams.data_val_unseen, device, hparams),
                             shuffle=False,
                             batch_size=hparams.val_unseen_size,
                             collate_fn=collate_fn),
    }

    model = Tacotron2(hparams).to(device)
    if hparams.speaker_encoder:
        model.speaker_encoder.load_state_dict(torch.load(hparams.speaker_encoder, map_location='cpu')['state_dict'])
        model.speaker_encoder.to(device)
        for param in model.speaker_encoder.parameters():
            param.requires_grad = False

    criterion = Tacotron2Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.learning_rate, weight_decay=hparams.weight_decay)

    logger = Tacotron2Logger(output_directory)

    return train_loader, val_loaders, model, criterion, optimizer, logger


def prepare_speaker_encoder(device, output_directory, hparams):
    train_loader = DataLoader(MelFragmentDataset(hparams.data_train, hparams.batch_size_speakers,
                                                 hparams.batch_size_speaker_samples, device, hparams), batch_size=None)
    val_loaders = {
        'seen': DataLoader(MelFragmentDataset(hparams.data_val_seen, hparams.val_seen_size_speakers,
                                              hparams.val_seen_size_speaker_samples, device, hparams),
                           batch_size=None),
        'unseen': DataLoader(MelFragmentDataset(hparams.data_val_unseen, hparams.val_unseen_size_speakers,
                                                hparams.val_unseen_size_speaker_samples, device, hparams),
                             batch_size=None),
    }

    model = SpeakerEncoder(hparams).to(device)
    criterion = SpeakerEncoderLoss().to(device)
    optimizer = torch.optim.SGD(chain(model.parameters(), criterion.parameters()), lr=hparams.learning_rate)

    logger = SpeakerEncoderLogger(output_directory)

    return train_loader, val_loaders, model, criterion, optimizer, logger


def validate(model, val_loaders, criterion, iteration, logger):
    model.eval()
    criterion.eval()

    params = {}
    with torch.no_grad():
        for name, val_loader in val_loaders.items():
            x, y = next(iter(val_loader))
            y_pred = model(x)
            val_loss = criterion(y_pred, y).item()
            params[name] = {'y': y, 'y_pred': y_pred, 'loss': val_loss}

    model.train()
    criterion.train()
    print('Validation loss {}: '.format(iteration) +
          ' '.join(['{} {:9f}'.format(key, key_params['loss']) for key, key_params in params.items()]))
    logger.log_validation(params, iteration)


def train(experiment, output_directory, checkpoint_path, warm_start, hparams):
    torch.manual_seed(hparams.seed)
    if hparams.use_cuda:
        torch.cuda.manual_seed(hparams.seed)

    device = torch.device('cuda' if hparams.use_cuda else 'cpu')
    prepare_fn = prepare_tacotron if experiment == 'tacotron' else prepare_speaker_encoder
    train_loader, val_loaders, model, criterion, optimizer, logger = prepare_fn(device, output_directory, hparams)

    learning_rate = hparams.learning_rate

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    if checkpoint_path is not None:
        if warm_start:
            model = warm_start_model(
                checkpoint_path, model, hparams.ignore_layers)
        else:
            model, critertion, optimizer, _learning_rate, iteration = \
                load_checkpoint(checkpoint_path, model, criterion, optimizer)
            iteration += 1  # next iteration is iteration + 1
            epoch_offset = max(0, int(iteration / len(train_loader)))

    model.train()
    criterion.train()
    is_overflow = False
    for epoch in range(epoch_offset, hparams.epochs):
        print(f'Epoch: {epoch}')
        for i, (x, y) in enumerate(train_loader):
            start = time.perf_counter()

            model.zero_grad()
            criterion.zero_grad()

            y_pred = model(x)
            loss = criterion(y_pred, y)

            loss.backward()
            loss = loss.item()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.grad_clip_thresh)

            optimizer.step()

            if not is_overflow:
                duration = time.perf_counter() - start
                print("Train loss {} {:.6f} Grad norm {:.6f} {:.2f}s/it".format(
                    iteration, loss, grad_norm, duration))
                logger.log_training(loss, duration, iteration, criterion)

            if not is_overflow and iteration % hparams.iters_per_checkpoint == 0:
                validate(model, val_loaders, criterion, iteration, logger)
                checkpoint_path = os.path.join(output_directory, f'checkpoint_{iteration}')
                save_checkpoint(model, criterion, optimizer, learning_rate, iteration, checkpoint_path)

            iteration += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='tacotron')
    parser.add_argument('-o', '--output', type=str, default=None)
    parser.add_argument('-c', '--checkpoint', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('--warm_start', action='store_true',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')

    args = parser.parse_args()

    model = args.model
    hparams = create_hparams(args.model, args.hparams)

    run_time = datetime.now().strftime("%b%d_%H_%M_%S")
    output_directory = run_time if args.output is None else f'{args.output}_{run_time}'
    output_directory = os.path.join('runs', output_directory)
    os.makedirs(output_directory)

    print(f'Model: {args.model}')
    print(f'Use CUDA: {hparams.use_cuda}')

    train(model, output_directory, args.checkpoint, args.warm_start, hparams)
