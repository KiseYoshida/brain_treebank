import logging
import numpy as np
import models
from torch.utils import data
import torch
from tasks import register_task
from tasks.base_regression_task import BaseRegressionTask
from tasks.batch_utils import baseline_wav_collator
from util.tensorboard_utils import plot_tensorboard_line
from sklearn.metrics import roc_auc_score, f1_score
from datasets import build_dataset
from tasks.utils import split_dataset


log = logging.getLogger(__name__)

@register_task(name="classification_regression_task")
class ClassificationRegressionTask(BaseRegressionTask):
    def __init__(self, cfg):
        super(ClassificationRegressionTask, self).__init__(cfg)
    
    def load_datasets(self, data_cfg, cached_seeg_data=None, cached_word_df=None,):
        #create train/val/test dataset
        dataset = build_dataset(data_cfg, task_cfg=self.cfg, cached_seeg_data=cached_seeg_data, cached_word_df=cached_word_df)
        train_X, train_y, test_X, test_y, val_X, val_y = split_dataset(dataset, data_cfg)
        self.train_X = train_X
        self.train_y = train_y

        self.test_X = test_X
        self.test_y = test_y

        self.val_X = val_X
        self.val_y = val_y

        self.dataset = dataset
