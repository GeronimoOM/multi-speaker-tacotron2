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
    def __init__(self, init_w=10.0, init_b=-5.0):
        super(SpeakerEncoderLoss, self).__init__()
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))

    def forward(self, speaker_vectors, speakers):
        N = len(speakers)
        M = len(speaker_vectors) // N

        C = []
        C_min = []
        for n in range(N):
            speaker_n_vectors = speaker_vectors[n * M:(n+1) * M]
            speaker_n_vectors_sum = speaker_n_vectors.sum(dim=0)
            C.append(speaker_n_vectors_sum / M)
            for m in range(M):
                C_min.append((speaker_n_vectors_sum - speaker_n_vectors[m]) / (M - 1))

        S = torch.empty(N, N * M, device=speaker_vectors.device)
        S_diag = F.cosine_similarity(torch.stack(C_min), speaker_vectors)
        for n in range(N):
            n_fr, n_to = n * M, (n + 1) * M
            S[n, :n_fr] = F.cosine_similarity(C[n].unsqueeze(0), speaker_vectors[:n_fr])
            S[n, n_fr:n_to] = S_diag[n_fr:n_to]
            S[n, n_to:] = F.cosine_similarity(C[n].unsqueeze(0), speaker_vectors[n_to:])
        L = -(self.w * S_diag + self.b) + (self.w * S + self.b).exp().sum(dim=0).log()

        return L.sum()
