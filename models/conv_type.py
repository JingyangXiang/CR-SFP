import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import distance


class SoftParameterMasker(autograd.Function):
    """Dynamic STE (straight-through estimator) parameter masker"""

    @staticmethod
    def forward(ctx, weight: torch.Tensor, mask: torch.Tensor):
        return weight * mask

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


class HardParameterMasker(autograd.Function):
    """Hard parameter masker"""

    @staticmethod
    def forward(ctx, weight: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(mask)
        return weight * mask

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        mask, = ctx.saved_tensors
        return grad_output * mask, None


# implemention for CR-SFP Conv2d
class SFPConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, padding_mode='zeros', **kwargs):
        super(SFPConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding,
                                        dilation, groups, bias, padding_mode, **kwargs)
        self.register_buffer("mask", torch.ones(self.out_channels, 1, 1, 1))
        self._mask_forward = False

    def forward(self, x):
        if self._mask_forward:
            w = HardParameterMasker.apply(self.weight, self.mask)
        else:
            w = self.weight
        x = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x

    @torch.no_grad()
    def init_mask(self, prune_rate, prune_criterion):
        if prune_criterion == 'l2':
            self.l2_mask(prune_rate)
        elif prune_criterion == 'taylor':
            self.taylor_mask(prune_rate)
        elif prune_criterion == 'fpgm':
            self.fpgm_mask(prune_rate)
        else:
            raise ValueError
        self.show_zero_num()

    @torch.no_grad()
    def l2_mask(self, prune_rate):
        filter_pruned_num = int(self.weight.shape[0] * prune_rate)
        weight_vec = self.weight.data.view(self.weight.shape[0], -1)
        score = torch.norm(weight_vec, 2, 1)
        value, index = torch.sort(score)
        mask = torch.zeros(score.shape, device=score.device)
        mask = mask.scatter_(dim=0, index=index[filter_pruned_num:], value=1.)
        mask = mask.reshape(-1, 1, 1, 1)
        self.mask.data = mask
        print("init mask with l2 norm", end=' | ')

    @torch.no_grad()
    def taylor_mask(self, prune_rate):
        if self.weight.grad is None:
            self.l2_mask(prune_rate)
        else:
            filter_pruned_num = int(self.weight.shape[0] * prune_rate)
            weight_vec = self.weight.data.view(self.weight.shape[0], -1)
            weight_vec_grad = self.weight.grad.view(self.weight.shape[0], -1)
            weight_vec = weight_vec * weight_vec_grad
            score = torch.abs(weight_vec).sum(dim=-1)
            value, index = torch.sort(score)
            mask = torch.zeros(score.shape, device=score.device)
            mask = mask.scatter_(dim=0, index=index[filter_pruned_num:], value=1.)
            mask = mask.reshape(-1, 1, 1, 1)
            self.mask.data = mask
            print("init mask with taylor norm", end=' | ')

    @torch.no_grad()
    def fpgm_mask(self, prune_rate):
        filter_pruned_num = int(self.weight.shape[0] * prune_rate)
        weight_vec_after_norm = self.weight.cpu().flatten(1).numpy()
        similar_matrix = distance.cdist(weight_vec_after_norm, weight_vec_after_norm, 'euclidean')
        score = torch.tensor(np.sum(np.abs(similar_matrix), axis=0), device=self.weight.device)
        value, index = torch.sort(score)
        mask = torch.zeros(score.shape, device=self.weight.device)
        mask = mask.scatter_(dim=0, index=index[filter_pruned_num:], value=1.)
        mask = mask.reshape(-1, 1, 1, 1)
        self.mask.data = mask
        print("init mask with fpgm norm", end=' | ')

    @torch.no_grad()
    def do_grad_mask(self):
        self.weight.grad.data = self.weight.grad.data * self.mask.data

    @torch.no_grad()
    def do_weight_mask(self):
        self.weight.data = self.weight.data * self.mask.data
        self.show_zero_num()

    def set_mask_forward_true(self):
        self._mask_forward = True

    def set_mask_forward_false(self):
        self._mask_forward = False

    @torch.no_grad()
    def show_zero_num(self):
        print(f"weight num: {self.weight.numel()}, zero num: {torch.sum(torch.eq(self.weight, 0))}")
