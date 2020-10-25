import numpy as np
from sklearn.metrics.pairwise import linear_kernel
from sklearn.decomposition import PCA
import random
from torch.utils.tensorboard import SummaryWriter
from plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy, \
     plot_speaker_embeddings_to_numpy


class Tacotron2Logger(SummaryWriter):

    def __init__(self, log_dir):
        super(Tacotron2Logger, self).__init__(log_dir)

    def log_training(self, loss, duration, iteration, _):
        self.add_scalar("training.loss", loss, iteration)
        self.add_scalar("duration", duration, iteration)

    def log_validation(self, params, iteration):
        for key, key_params in params.items():
            self.add_scalar(f'{key}.loss', key_params['loss'], iteration)

            _, mel_outputs, gate_outputs, alignments = key_params['y_pred']
            mel_targets, gate_targets = key_params['y']

            # plot alignment, mel target and predicted, gate target and predicted
            idx = random.randint(0, mel_outputs.size(0) - 1)
            self.add_image(
                f'{key}.alignment',
                plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
                iteration, dataformats='HWC')
            self.add_image(
                f'{key}.mel_target',
                plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy()),
                iteration, dataformats='HWC')
            self.add_image(
                f'{key}.mel_predicted',
                plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy()),
                iteration, dataformats='HWC')


class SpeakerEncoderLogger(SummaryWriter):

    def __init__(self, log_dir):
        super(SpeakerEncoderLogger, self).__init__(log_dir)

    def log_training(self, loss, duration, iteration, criterion):
        self.add_scalar("training.loss", loss, iteration)
        self.add_scalar("duration", duration, iteration)
        criterion_params = {name: param.item() for name, param in criterion.named_parameters()}
        self.add_scalar("loss.w", criterion_params['w'], iteration)
        self.add_scalar("loss.b", criterion_params['b'], iteration)

    def log_validation(self, params, iteration):
        for key, key_params in params.items():
            self.add_scalar(f'{key}.loss', key_params['loss'], iteration)
            y = key_params['y']
            y_pred = key_params['y_pred'].cpu().numpy()

            C, D = self.calc_CD(y_pred, y)
            self.add_scalar(f'{key}.c', C, iteration)
            self.add_scalar(f'{key}.d', D, iteration)

            pca = PCA(n_components=2)
            y_pred_2d = pca.fit_transform(y_pred)
            exp_var = pca.explained_variance_ratio_[:2].sum()
            self.add_scalar(f'{key}.exp_var', exp_var, iteration)
            self.add_image(
                f'{key}.embeddings',
                plot_speaker_embeddings_to_numpy(y_pred_2d, y),
                iteration, dataformats='HWC')

    def calc_CD(self, embeddings, speakers):
        N = len(speakers)
        M = len(embeddings) // N
        S = linear_kernel(embeddings)

        C = []  # N * MxM
        D = []  # N * M(N-1)xM
        for j in range(N):
            C.append(S[j * M:(j + 1) * M, j * M:(j + 1) * M])
            D.append(np.hstack([S[j * M:(j + 1) * M, :j * M], S[j * M:(j + 1) * M, (j + 1) * M:]]))

        return np.mean(C), np.mean(D)
