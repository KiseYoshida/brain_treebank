import random
import copy
from tqdm.contrib.logging import logging_redirect_tqdm
import numpy as np
import torch.nn as nn
import os
from tqdm import tqdm
import torch
import tasks
import torch.multiprocessing as mp
import torch.distributed as dist
import logging
from tensorboardX import SummaryWriter
from schedulers import build_scheduler
import torch_optimizer as torch_optim
from sklearn import linear_model
from sklearn.model_selection import cross_val_score, cross_validate
import scipy
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score

log = logging.getLogger(__name__)

class RegressionRunnerTestSet():
    def __init__(self, cfg, task, model, criterion):
        self.cfg = cfg
        self.model = model#NOTE: not used
        self.task = task
        self.criterion = criterion#NOTE: not used
        self.exp_dir = os.getcwd()

    def split_and_train_and_test(self, count):
        #split the data and then train on it and then test on test

        X_train, y_train = self.task.train_X, self.task.train_y
        X_test, y_test = self.task.test_X, self.task.test_y
        #X_val, y_val = self.task.val_X, self.task.val_y

        X_train = scipy.signal.decimate(X_train, 10, axis=-1)
        #X_val = scipy.signal.decimate(X_val, 10, axis=-1)
        X_test = scipy.signal.decimate(X_test, 10, axis=-1)

        model = linear_model.LogisticRegression(random_state=0, max_iter=500)
        model = make_pipeline(preprocessing.StandardScaler(), model)
        cv_score = cross_val_score(model, X_train, y_train, cv=self.task.cfg.num_repeat, scoring="roc_auc")

        model = linear_model.LogisticRegression(random_state=0, max_iter=500)
        model = make_pipeline(preprocessing.StandardScaler(), model)
        model.fit(X_train, y_train)

        #train_score = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
        test_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        #val_score = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

        #if count >= 10:
        #    import pdb; pdb.set_trace()
        cv_score = cv_score.mean()
        #"val_score": val_score,
        return {"test_score": test_score, "cv_score": cv_score}
        #"cv_score": cv_score}

class RegressionRunnerControlledTestSet():
    def __init__(self, cfg, task, model, criterion):
        self.cfg = cfg
        self.model = model#NOTE: not used
        self.task = task
        self.criterion = criterion#NOTE: not used
        self.exp_dir = os.getcwd()

    def split_and_train_and_test(self):
        #split the data and then train on it and then test on test
        results = []
        for i in range(10):#TODO flag
            X = self.task.dataset.seeg_data
            X = scipy.signal.decimate(X, 10, axis=-1)
            X = scipy.stats.zscore(X)
            y = np.array(self.task.dataset.label)
            word_df = self.task.dataset.word_df #consider moving this to regression_task.py
            
            #equalize test_set
            test_size = int(0.1*len(X))#TODO flag
            all_idxs = np.arange(X.shape[0])
            control_test_set = False#TODO flag
            if control_test_set:
                controls = ["is_onset", "is_midset", "is_offset"]
                n_targets = 2 #for pos
                control_size = int(test_size/(len(controls)*n_targets))
                all_control_idxs = []
                for control in controls:
                    control_idxs = word_df[word_df[control]==1].index
                    target = "pos"#TODO make this general
                    target_1_idxs = word_df[word_df[target]=="NOUN"].index
                    target_2_idxs = word_df[word_df[target]=="VERB"].index

                    target_1_control_idxs = np.intersect1d(target_1_idxs, control_idxs)
                    target_2_control_idxs = np.intersect1d(target_2_idxs, control_idxs)
                    assert len(target_1_control_idxs) > control_size and len(target_2_control_idxs) > control_size

                    all_control_idxs += random.sample(list(target_1_control_idxs), control_size)
                    all_control_idxs += random.sample(list(target_2_control_idxs), control_size)

                test_idxs = all_control_idxs
            else:
                test_idxs = random.sample(list(all_idxs), test_size)

            train_idxs = np.setdiff1d(all_idxs, test_idxs)
            X_train, y_train = X[train_idxs], y[train_idxs]
            X_test, y_test = X[test_idxs], y[test_idxs]
            model = linear_model.LogisticRegression(random_state=0, max_iter=500)
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            results.append(score)
        return results
 
class RegressionRunner():
    def __init__(self, cfg, task, model, criterion):
        self.cfg = cfg
        self.model = model#NOTE: not used
        self.task = task
        self.criterion = criterion#NOTE: not used
        self.exp_dir = os.getcwd()

    def cross_validate(self):
        X = self.task.dataset.seeg_data
        X = scipy.signal.decimate(X, 10, axis=-1)
        #X = scipy.stats.zscore(X)

        y = np.array(self.task.dataset.label)
        model = linear_model.LogisticRegression(random_state=0, max_iter=500)
        clf = make_pipeline(preprocessing.StandardScaler(), model)
        score = cross_val_score(clf, X, y, cv=self.task.cfg.num_repeat, scoring="roc_auc")

        #clf = make_pipeline(preprocessing.StandardScaler(), model)
        #score_dict = cross_validate(clf, X, y, cv=self.task.cfg.num_repeat, scoring="roc_auc", return_train_score=True)
        return score
