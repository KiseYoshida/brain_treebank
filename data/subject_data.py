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

class SubjectData():
    def __init__(self, cfg) -> None:
        self.selected_electrodes = cfg.electrodes
        self.selected_words = cfg.words
        self.cfg = cfg
        self.words, self.neural_data, self.trials = self.get_subj_data(cfg.subject)

    def get_subj_data(self, subject):
        words, seeg_data, trials = [], [], []
        cached_transcript_aligns = self.cfg.cached_transcript_aligns
        for trial in self.cfg.brain_runs:
            trial_cfg = self.cfg.copy()
            if cached_transcript_aligns: #TODO: I want to make this automatic
                trial_cached_transcript_aligns = os.path.join(cached_transcript_aligns, subject, trial)
                os.makedirs(trial_cached_transcript_aligns, exist_ok=True)
                trial_cfg.cached_transcript_aligns = trial_cached_transcript_aligns
            t = TrialData(subject, trial, trial_cfg)
            reader = TrialDataReader(t, trial_cfg)

            trial_words, seeg_trial_data = reader.get_aligned_predictor_matrix(duration=self.cfg.duration, delta=self.cfg.delta)
            assert (range(seeg_trial_data.shape[1]) == trial_words.index).all()
            trial_words['movie_id'] = t.movie_id
            trials.append(t)
            words.append(trial_words)
            seeg_data.append(seeg_trial_data)

        neural_data = np.concatenate(seeg_data, axis=1)
        #neural_data is [n_electrodes, n_words, n_samples]
        words_df = pd.concat(words) #NOTE the index will not be unique, but the location will
        words_df["is_midset"] = (words_df["is_onset"]==0) & (words_df["is_offset"])==0
        words_df = words_df.reset_index() #now the index will be unique
        if not np.all(words_df.index == np.arange(len(words_df))):
            import pdb; pdb.set_trace()
        words_df["is_midset"] = (words_df["is_onset"]==0) & (words_df["is_offset"]==0)
        assert np.all(words_df.index == np.arange(len(words_df)))
        return words_df, neural_data, trials
