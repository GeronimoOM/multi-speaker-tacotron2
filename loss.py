import torch
from torch import nn
import torch.nn.functional as F


class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _ = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        return mel_loss + gate_loss


class SpeakerEncoderLoss(nn.Module):
    def __init__(self, batch_size_speakers, batch_size_speaker_samples, alpha=0.66):
        super(SpeakerEncoderLoss, self).__init__()
        self.N = batch_size_speakers
        self.M = batch_size_speaker_samples
        self.alpha = alpha

    def forward(self, speaker_vectors, _):
        C = []
        C_min = []
        for n in range(self.N):
            speaker_n_vectors = speaker_vectors[n*self.M:(n+1)*self.M]
            speaker_n_vectors_sum = speaker_n_vectors.sum(dim=0)
            C.append(speaker_n_vectors_sum / self.M)
            for m in range(self.M):
                C_min.append((speaker_n_vectors_sum - speaker_n_vectors[m]) / (self.M - 1))

        S = torch.empty(self.N, self.N*self.M, device=speaker_vectors.device)
        S_diag = F.cosine_similarity(torch.stack(C_min), speaker_vectors)
        for n in range(self.N):
            n_fr, n_to = n * self.M, (n + 1) * self.M
            S[n, :n_fr] = F.cosine_similarity(C[n].unsqueeze(0), speaker_vectors[:n_fr])
            S[n, n_fr:n_to] = S_diag[n_fr:n_to]
            S[n, n_to:] = F.cosine_similarity(C[n].unsqueeze(0), speaker_vectors[n_to:])
        L = -S_diag + self.alpha * S.exp().sum(dim=0).log()

        return L.sum()
