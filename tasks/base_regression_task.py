import models
import criterions
from torch.utils import data
import torch
from datasets import build_dataset
from tasks.utils import split_dataset

class BaseRegressionTask():
    def __init__(self, cfg):
        self.cfg = cfg

    def build_model(self, cfg):
        return models.build_model(cfg)

    def load_datasets(self, data_cfg, cached_seeg_data=None, cached_word_df=None,):
        #create train/val/test dataset
        dataset = build_dataset(data_cfg, task_cfg=self.cfg, cached_seeg_data=cached_seeg_data, cached_word_df=cached_word_df)
        self.dataset = dataset

    @classmethod
    def setup_task(cls, cfg):
        return cls(cfg)

    def build_criterion(self, cfg):
        return criterions.build_criterion(cfg)

