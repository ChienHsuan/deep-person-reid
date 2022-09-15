import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    def __init__(self, use_gpu=True, m=0.5, s=64, label_smooth=True,
                 epsilon=0.1):
        super(ArcFaceLoss, self).__init__()
        self.use_gpu = use_gpu
        self.label_smooth = label_smooth
        self.epsilon = epsilon

        assert m >= 0
        self.class_margins = m
        assert s > 0
        self.s = s

    def forward(self, cos_theta, target):
        """
        Args:
            cos_theta (torch.Tensor): prediction matrix (before softmax) with
                shape (batch_size, num_classes).
            target (torch.LongTensor): ground truth labels with shape (batch_size).
                Each position contains the label index.
        """

        one_hot_target = torch.zeros_like(cos_theta, dtype=torch.uint8)
        one_hot_target.scatter_(1, target.data.view(-1, 1), 1)
        self.class_margins *= one_hot_target

        output = self.s * torch.cos(torch.acos(cos_theta) + self.class_margins)

        if self.label_smooth:
            target = torch.zeros(output.size(), device=target.device).scatter_(1, target.detach().unsqueeze(1), 1)
            num_classes = output.size(1)
            target = (1.0 - self.epsilon) * target + self.epsilon / float(num_classes)
            losses = (- target * F.log_softmax(output, dim=1)).sum(dim=1)

        else:
            losses = F.cross_entropy(output, target, reduction='none')

        return losses.mean()