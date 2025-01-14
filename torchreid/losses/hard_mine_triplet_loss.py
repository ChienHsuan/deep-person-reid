import torch
import torch.nn as nn
import torch.nn.functional as F


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Tensor, with shape [m, d]
      y: pytorch Tensor, with shape [n, d]
    Returns:
      dist: pytorch Tensor, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def cosine_dist(x, y):
    """
    Args:
      x: pytorch Tensor, with shape [m, d]
      y: pytorch Tensor, with shape [n, d]
    """
    x_normed = F.normalize(x, p=2, dim=1)
    y_normed = F.normalize(y, p=2, dim=1)
    return 1 - torch.mm(x_normed, y_normed.t())

def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Tensor, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Tensor, distance(anchor, positive); shape [N]
      dist_an: pytorch Tensor, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())
    
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        ind = (labels.new().resize_as_(labels)
    		   .copy_(torch.arange(0, N).long())
        	   .unsqueeze(0).expand(N, N))
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


# ==============
#  Triplet Loss 
# ==============
class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    """
    def __init__(self, margin=None, metric="cosine"):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.metric = metric
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, global_feat, labels):
        if self.metric == "euclidean":
            dist_mat = euclidean_dist(global_feat, global_feat)
        elif self.metric == "cosine":
            dist_mat = cosine_dist(global_feat, global_feat)
        else:
            raise NameError

        dist_ap, dist_an = hard_example_mining(dist_mat, labels)
        y = dist_an.new_ones(dist_an.size())
        
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss
