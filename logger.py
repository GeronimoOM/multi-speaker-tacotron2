import numpy as np
from sklearn.decomposition import PCA
import random
from torch.utils.tensorboard import SummaryWriter
from plotting_utils import plot_speaker_embeddings_to_numpy


class Tacotron2Logger(SummaryWriter):

    def __init__(self, log_dir):
        super(Tacotron2Logger, self).__init__(log_dir)

    def log_training(self, loss, grad_norm, learning_rate, duration, iteration, criterion):
        self.add_scalar("training.loss", loss, iteration)
        self.add_scalar("grad.norm", grad_norm, iteration)
        self.add_scalar("learning.rate", learning_rate, iteration)
        self.add_scalar("duration", duration, iteration)

    def log_validation(self, loss, y, y_pred, iteration):
        self.add_scalar("validation.loss", loss, iteration)
        _, mel_outputs, gate_outputs, alignments = y_pred
        mel_targets, gate_targets = y


        # plot alignment, mel target and predicted, gate target and predicted
        idx = random.randint(0, alignments.size(0) - 1)
        '''self.add_image(
            "alignment",
            plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
            iteration, dataformats='HWC')
        self.add_image(
            "mel_target",
            plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy()),
            iteration, dataformats='HWC')
        self.add_image(
            "mel_predicted",
            plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy()),
            iteration, dataformats='HWC')
        self.add_image(
            "gate",
            plot_gate_outputs_to_numpy(
                gate_targets[idx].data.cpu().numpy(),
                torch.sigmoid(gate_outputs[idx]).data.cpu().numpy()),
            iteration, dataformats='HWC')
            '''


class SpeakerEncoderLogger(SummaryWriter):

    def __init__(self, log_dir):
        super(SpeakerEncoderLogger, self).__init__(log_dir)

    def log_training(self, loss, grad_norm, learning_rate, duration, iteration, criterion):
        self.add_scalar("training.loss", loss, iteration)
        self.add_scalar("grad.norm", grad_norm, iteration)
        self.add_scalar("learning.rate", learning_rate, iteration)
        self.add_scalar("duration", duration, iteration)
        criterion_params = {name: param.item() for name, param in criterion.named_parameters()}
        self.add_scalar("loss.w", criterion_params['w'], iteration)
        self.add_scalar("loss.b", criterion_params['b'], iteration)

    def log_validation(self, loss, y, y_pred, iteration):
        self.add_scalar("validation.loss", loss, iteration)

        pca = PCA(n_components=2)
        y_pred_2d = pca.fit_transform(y_pred.cpu().numpy())
        self.add_image(
            'speaker embeddings',
            plot_speaker_embeddings_to_numpy(y_pred_2d, y.cpu().numpy()),
            iteration, dataformats='HWC')


