import os
import time
import argparse
import math

import torch
from torch.utils.data import DataLoader

from model import Tacotron2, SpeakerEncoder
from data_utils import TextMelLoader, TextMelCollate, MelFragmentLoader
from loss import Tacotron2Loss, SpeakerEncoderLoss
from hparams import create_hparams


def prepare_directories_and_logger(output_directory, log_directory):
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    #return Tacotron2Logger(os.path.join(output_directory, log_directory))


def prepare_tacotron(hparams):
    trainset = TextMelLoader(hparams.training_files, hparams)
    valset = TextMelLoader(hparams.validation_files, hparams)

    collate_fn = TextMelCollate(hparams.n_frames_per_step)

    train_loader = DataLoader(trainset, shuffle=True,
                              batch_size=hparams.batch_size, pin_memory=True,
                              drop_last=True, collate_fn=collate_fn)

    val_loader = DataLoader(valset, shuffle=False, batch_size=hparams.batch_size,
                            pin_memory=True, collate_fn=collate_fn)

    model = Tacotron2(hparams)
    criterion = Tacotron2Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.learning_rate, weight_decay=hparams.weight_decay)

    return train_loader, val_loader, model, criterion, optimizer


def prepare_speaker_encoder(hparams):
    trainset = MelFragmentLoader(hparams.training_files, hparams)
    valset = MelFragmentLoader(hparams.validation_files, hparams)

    train_loader = DataLoader(trainset, batch_size=None)
    val_loader = DataLoader(valset, batch_size=None)

    model = SpeakerEncoder(hparams)
    criterion = SpeakerEncoderLoss(hparams.batch_size_speakers, hparams.batch_size_speaker_samples)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.learning_rate, weight_decay=hparams.weight_decay)

    return train_loader, val_loader, model, criterion, optimizer


def warm_start_model(checkpoint_path, model, ignore_layers=None):
    assert os.path.isfile(checkpoint_path)
    print(f'Warm starting model from checkpoint ${checkpoint_path}')
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = checkpoint_dict['state_dict']
    if ignore_layers is not None:
        model_dict = {k: v for k, v in model_dict.items()
                      if k not in ignore_layers}
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    model.load_state_dict(model_dict)
    return model


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print(f'Loading checkpoint ${checkpoint_path}')
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print(f'Loaded checkpoint ${checkpoint_path} from iteration ${iteration}')
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print(f'Saving model and optimizer state at iteration ${iteration} to ${filepath}')
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)


def validate(model, val_loader, criterion, iteration, logger):
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for i, batch in enumerate(val_loader):
            x, y = model.parse_batch(batch)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            val_loss += loss.item()
        val_loss = val_loss / (i + 1)

    model.train()
    print("Validation loss {}: {:9f}  ".format(iteration, val_loss))
    logger.log_validation(val_loss, model, y, y_pred, iteration)


def train(model, output_directory, log_directory, checkpoint_path, warm_start, hparams):
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    prepare_fn = prepare_tacotron if model == 'tacotron' else prepare_speaker_encoder
    train_loader, val_loader, model, criterion, optimizer = prepare_fn(hparams)
    device = torch.device('cuda' if hparams.use_cuda else 'cpu')
    model.to(device)

    learning_rate = hparams.learning_rate

    if hparams.fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level='O2')

    logger = None # prepare_directories_and_logger(output_directory, log_directory)

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
        print("Epoch: {}".format(epoch))
        for i, (x, y) in enumerate(train_loader):
            start = time.perf_counter()
            for param_group in optimizer.param_groups:
                param_group['lr'] = hparams.learning_rate

            model.zero_grad()
            y_pred = model(x)

            loss = criterion(y_pred, y)
            if hparams.fp16_run:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if hparams.fp16_run:
                grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), hparams.grad_clip_thresh)
                is_overflow = math.isnan(grad_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), hparams.grad_clip_thresh)

            optimizer.step()

            if not is_overflow:
                duration = time.perf_counter() - start
                print("Train loss {} {:.6f} Grad Norm {:.6f} {:.2f}s/it".format(
                    iteration, loss.item(), grad_norm, duration))
                logger.log_training(loss.item(), grad_norm, learning_rate, duration, iteration)

            if not is_overflow and (iteration % hparams.iters_per_checkpoint == 0):
                validate(model, val_loader, criterion, iteration, logger)
                checkpoint_path = os.path.join(
                    output_directory, "checkpoint_{}".format(iteration))
                save_checkpoint(model, optimizer, learning_rate, iteration,
                                checkpoint_path)

            iteration += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='tacotron')
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save checkpoints')
    parser.add_argument('-l', '--log_directory', type=str,
                        help='directory to save tensorboard logs')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('--warm_start', action='store_true',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')

    args = parser.parse_args()
    hparams = create_hparams(args.model, args.hparams)

    print("FP16 Run:", hparams.fp16_run)
    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    print("Use CUDA :", hparams.use_cuda)

    train(args.model, args.output_directory, args.log_directory, args.checkpoint_path, args.warm_start, hparams)
