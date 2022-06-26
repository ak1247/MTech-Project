import torch
from torch import nn
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")

##
def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

## Inv scheduler
def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    # print(param_group['lr'])
    return optimizer

def inv_lr_scheduler(param_lr, optimizer, iter_num, gamma=10, power=0.75, init_lr=0.001, weight_decay=0.0005,max_iter=10000):
    gamma = 10.0
    lr = init_lr * (1 + gamma * min(1.0, iter_num / max_iter)) ** (-power)
    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        i += 1
    print( param_group['lr'])
    return optimizer


### Loss -functions

def entropyLoss(p):
    p = F.softmax(p)
    return -torch.mean(torch.sum(p * torch.log(p + 1e-5), 1))

def wt_entropy_loss(p,wt):
    p = F.softmax(p)
    return -torch.mean(torch.sum(p * torch.log(p + 1e-5), 1)*wt)

def cross_entropy_loss(p1,p2):
    p1 = F.softmax(p1)
    p2 = F.softmax(p2)
    return -torch.mean(torch.sum(p1 * torch.log(p2 + 1e-5), 1))

def div_loss(p):
    p = F.softmax(p)
    msoftmax = p.mean(dim=0)
    gentropy_loss = torch.sum(msoftmax * torch.log(msoftmax + 1e-5))
    return gentropy_loss

def locDiv_loss(p,n,temp):
    p = F.softmax(p)
    msoftmax = p.mean(dim=0)
    flatten = msoftmax.pow(temp)/msoftmax.pow(temp).sum()
    return F.kl_div(msoftmax,flatten)

def uniEnt_loss(p,n):
    p = F.softmax(p)
    return -torch.mean(torch.sum(torch.log(p+1e-5),1)/n)


## CE with label smoothing
class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss
        return loss