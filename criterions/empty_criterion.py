import torch
from .base_criterion import BaseCriterion
from torch import nn
from criterions import register_criterion

@register_criterion("empty_criterion")
class EmptyCriterion(BaseCriterion):
    def __init__(self):
        super(EmptyCriterion, self).__init__()
        pass

    def build_criterion(self, cfg):
        self.cfg = cfg

