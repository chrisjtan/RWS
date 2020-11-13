from torch import nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class TripletLossWithConfidence(nn.Module):
    """
    Triplet loss with confidence c between 0 to 1
    Takes embeddings of an anchor sample, a positive sample and a negative sample with the confidence of these samples
    being positive or negative.
    """

    def __init__(self, margin):
        super(TripletLossWithConfidence, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, c, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + c * self.margin)
        return losses.mean() if size_average else losses.sum()