#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# @Author: xyshu
# @Time: 2023-08-06 10:36
# @File: quantile_loss.py
# @Description: implementation of TFT model

import torch
import torch.nn as nn


class QuantileLoss(nn.Module):
    """
    Implementation source: https://medium.com/the-artificial-impostor/quantile-regression-part-2-6fdbc26b2629

    Args:
          quantiles (list): List of quantiles that will be used for prediction
    """

    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        """
        Args:
              preds (torch.tensor): Model predictions
              target (torch.tensor): Target data
        """
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)

        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))

        loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))

        return loss
