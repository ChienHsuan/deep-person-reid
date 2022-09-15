import torch
import torch.nn as nn
import torch.nn.functional as F


class AMSoftmaxLoss(nn.Module):
    def __init__(self, use_gpu=True, m=0.35, s=30.0, label_smooth=True,
                 epsilon=0.1, conf_penalty=0.3, pr_product=False):
        super(AMSoftmaxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.label_smooth = label_smooth
        self.epsilon = epsilon
        self.conf_penalty = conf_penalty
        self.pr_product = pr_product

        self.ce = nn.CrossEntropyLoss()

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

        if self.pr_product:
            pr_alpha = torch.sqrt(1.0 - cos_theta.pow(2.0))
            cos_theta = pr_alpha.detach() * cos_theta + cos_theta.detach() * (1.0 - pr_alpha)

        one_hot_target = torch.zeros_like(cos_theta, dtype=torch.uint8)
        one_hot_target.scatter_(1, target.data.view(-1, 1), 1)
        self.class_margins *= one_hot_target

        phi_theta = cos_theta - self.class_margins
        phi_theta *= self.s

        if self.label_smooth:
            target = torch.zeros(phi_theta.size(), device=target.device).scatter_(1, target.detach().unsqueeze(1), 1)
            num_classes = phi_theta.size(1)
            target = (1.0 - self.epsilon) * target + self.epsilon / float(num_classes)
            losses = (- target * F.log_softmax(phi_theta, dim=1)).sum(dim=1)

        else:
            losses = F.cross_entropy(phi_theta, target, reduction='none')

        if self.conf_penalty > 0.0:
            probs = F.softmax(phi_theta, dim=1)
            log_probs = F.log_softmax(phi_theta, dim=1)
            entropy = torch.sum(-probs * log_probs, dim=1)
            losses = F.relu(losses - self.conf_penalty * entropy)

        with torch.no_grad():
            nonzero_count = max(losses.nonzero().size(0), 1)

        return losses.sum() / nonzero_count
        