import time
import os
import torch
import string
import numpy as np
import h5py
# import numpy.typing as npt

from torch.utils import data
from .trial_data import TrialData
from .trial_data_reader import TrialDataReader
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
from types import SimpleNamespace

class GLMSubjectData():
    def __init__(self, cfg) -> None:
        self.selected_electrodes = cfg.electrodes
        self.selected_words = cfg.words
        self.cfg = cfg
        self.words, self.neural_data, self.trials = self.get_subj_data(cfg.subject)

    def get_subj_data(self, subject):
        words, seeg_data, trials = [], [], []
        for trial in self.cfg.brain_runs:
            trial_cfg = self.cfg.copy()
            cached_transcript_aligns = self.cfg.get("cached_transcript_aligns", None)
            if cached_transcript_aligns: #TODO: I want to make this automatic
                cached_transcript_aligns = os.path.join(cached_transcript_aligns, subject, trial)
                os.makedirs(cached_transcript_aligns, exist_ok=True)
                trial_cfg.cached_transcript_aligns = cached_transcript_aligns
            t = TrialData(subject, trial, trial_cfg)
            reader = TrialDataReader(t, trial_cfg)

            trial_words, seeg_trial_data = reader.get_aligned_predictor_matrix(duration=self.cfg.duration, delta=self.cfg.delta)
            trial_words["is_midset"] = (trial_words["is_onset"]==0) & (trial_words["is_offset"])==0
            assert (range(seeg_trial_data.shape[1]) == trial_words.index).all()
            trial_words['movie_id'] = t.movie_id
            #trial_words["trial"] = int(trial[5:])
            trials.append(t)
            words.append(trial_words)
            seeg_data.append(seeg_trial_data)

        neural_data = seeg_data
        words_df = words
        return words_df, neural_data, trials

