import os
import time
import argparse
import math
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from model import Tacotron2, SpeakerEncoder
from data_utils import TextMelDataset, TextMelCollate, MelFragmentDataset
from loss import Tacotron2Loss, SpeakerEncoderLoss
from logger import Tacotron2Logger, SpeakerEncoderLogger
from train_utils import warm_start_model, load_checkpoint, save_checkpoint
from hparams import create_hparams
from numpy import finfo


def prepare_tacotron(device, output_directory, hparams):
    trainset = TextMelDataset(hparams.training_files, device, hparams)
    valset = TextMelDataset(hparams.validation_files, device, hparams)

    collate_fn = TextMelCollate(hparams.n_frames_per_step)

    train_loader = DataLoader(trainset, shuffle=True,
                              batch_size=hparams.batch_size, pin_memory=True,
                              drop_last=True, collate_fn=collate_fn)

    val_loader = DataLoader(valset, shuffle=False, batch_size=hparams.batch_size,
                            pin_memory=True, collate_fn=collate_fn)

    model = Tacotron2(hparams).to(device)
    if hparams.fp16_run:
        model.decoder.attention_layer.score_mask_value = finfo('float16').min

    criterion = Tacotron2Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.learning_rate, weight_decay=hparams.weight_decay)

    logger = Tacotron2Logger(output_directory)

    return train_loader, val_loader, model, criterion, optimizer, logger


def prepare_speaker_encoder(device, output_directory, hparams):
    trainset = MelFragmentDataset(hparams.training_files, device, hparams)
    valset = MelFragmentDataset(hparams.validation_files, device, hparams)

    train_loader = DataLoader(trainset, batch_size=None)
    val_loader = DataLoader(valset, batch_size=None)

    model = SpeakerEncoder(hparams).to(device)
    criterion = SpeakerEncoderLoss(hparams.batch_size_speakers, hparams.batch_size_speaker_samples)
    optimizer = torch.optim.SGD(model.parameters(), lr=hparams.learning_rate)

    logger = SpeakerEncoderLogger(output_directory)

    return train_loader, val_loader, model, criterion, optimizer, logger


def validate(model, val_loader, criterion, iteration, logger):
    model.eval()
    ys = []
    ys_pred = []
    with torch.no_grad():
        val_loss = 0.0
        for i, (x, y) in enumerate(val_loader):
            y_pred = model(x)
            loss = criterion(y_pred, y)
            val_loss += loss.item()
            ys.append(y)
            ys_pred.append(y_pred)
        val_loss = val_loss / (i + 1)

    ys = torch.cat(ys)
    ys_pred = torch.cat(ys_pred)

    model.train()
    print("Validation loss {}: {:9f}  ".format(iteration, val_loss))
    logger.log_validation(val_loss, model, ys, ys_pred, iteration)


def train(model, output_directory, checkpoint_path, warm_start, hparams):
    torch.manual_seed(hparams.seed)

    device = torch.device('cuda' if hparams.use_cuda else 'cpu')
    prepare_fn = prepare_tacotron if model == 'tacotron' else prepare_speaker_encoder
    train_loader, val_loader, model, criterion, optimizer, logger = prepare_fn(device, output_directory, hparams)

    learning_rate = hparams.learning_rate

    if hparams.fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level='O2')

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    if checkpoint_path is not None:
        if warm_start:
            model = warm_start_model(checkpoint_path, model, hparams.ignore_layers)
        else:
            model, optimizer, _learning_rate, iteration = load_checkpoint(checkpoint_path, model, optimizer)
            iteration += 1  # next iteration is iteration + 1
            epoch_offset = max(0, int(iteration / len(train_loader)))

    model.train()
    is_overflow = False
    for epoch in range(epoch_offset, hparams.epochs):
        print(f'Epoch: {epoch}')
        for i, (x, y) in enumerate(train_loader):
            start = time.perf_counter()

            model.zero_grad()
            y_pred = model(x)

            loss = criterion(y_pred, y)
            if hparams.fp16_run:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            loss = loss.item()

            if hparams.fp16_run:
                grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), hparams.grad_clip_thresh)
                is_overflow = math.isnan(grad_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.grad_clip_thresh)

            optimizer.step()

            if not is_overflow:
                duration = time.perf_counter() - start
                print("Train loss {} {:.6f} Grad norm {:.6f} {:.2f}s/it".format(
                    iteration, loss, grad_norm, duration))
                logger.log_training(loss, grad_norm, learning_rate, duration, iteration)

            if not is_overflow and (iteration % hparams.iters_per_checkpoint == 0):
                validate(model, val_loader, criterion, iteration, logger)
                checkpoint_path = os.path.join(output_directory, f'checkpoint_{iteration}')
                save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path)

            iteration += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='tacotron')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('--warm_start', action='store_true',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')

    args = parser.parse_args()
    run_time = datetime.now().strftime("%b%d_%H_%M_%S")
    output_directory = os.path.join('runs', run_time)
    os.makedirs(output_directory)

    model = args.model
    hparams = create_hparams(args.model, args.hparams)

    print(f'Model: {args.model}')
    print(f'Use CUDA: {hparams.use_cuda}')
    print(f'FP16 Run: {hparams.fp16_run}')

    train(model, output_directory, args.checkpoint_path, args.warm_start, hparams)
