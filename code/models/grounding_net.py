import torch
import torch.nn as nn
import torch.nn.functional as F

class LBSign(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class Grounding(nn.Module):
    def __init__(self):
        super(Grounding, self).__init__()

        # visual net dimension reduction
        self.fc_v1 = nn.Linear(512, 128)
        self.fc_v2 = nn.Linear(128, 128)

        # audio net dimension reduction
        self.fc_a1 = nn.Linear(512, 128)
        self.fc_a2 = nn.Linear(128, 128)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)

        self.relu = nn.ReLU()
        self.bn = LBSign.apply
        self.sigmoid = nn.Sigmoid()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, feat_sound, feat_img):

        feat = torch.cat((feat_sound,  feat_img), dim =-1)
        g = self.fc3(self.relu(self.fc2(self.relu(self.fc1(feat)))))
        g = g*2
        return g