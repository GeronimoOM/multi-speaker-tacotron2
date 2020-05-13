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
    def __init__(self, batch_size_speakers, batch_size_speaker_samples, init_w=10.0, init_b=-5.0):
        super(SpeakerEncoderLoss, self).__init__()
        self.N = batch_size_speakers
        self.M = batch_size_speaker_samples
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))

    def forward(self, speaker_vectors, _):
        centroids = []
        centroids_without_one = []
        for n in range(self.N):
            speaker_n_vectors = speaker_vectors[n*self.M:(n+1)*self.M]
            speaker_n_vectors_sum = speaker_n_vectors.sum(dim=0)
            centroids.append(speaker_n_vectors_sum / self.M)
            for m in range(self.M):
                centroids_without_one.append((speaker_n_vectors_sum - speaker_n_vectors[m]) / (self.M - 1))

        S = torch.empty(self.N, self.N*self.M)
        S_same_centroid = self.w * F.cosine_similarity(torch.stack(centroids_without_one), speaker_vectors) + self.b
        for n in range(self.N):
            S[n] = self.w * F.cosine_similarity(centroids[n].unsqueeze(0), speaker_vectors) + self.b
            S[n, n*self.M:(n+1)*self.M] = S_same_centroid[n*self.M:(n+1)*self.M]
        L = -S_same_centroid + S.exp().sum(dim=0).log()

        return L.sum()
