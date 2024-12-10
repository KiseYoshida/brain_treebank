import random
import os
import torch
from tqdm import tqdm as tqdm
import numpy as np
from omegaconf import DictConfig, OmegaConf
from torch.utils import data
from data.subject_data import SubjectData
from data.electrode_subject_data import ElectrodeSubjectData
from data.speech_nonspeech_subject_data import NonLinguisticSubjectData, SentenceOnsetSubjectData
from data.timestamped_subject_data import TimestampedSubjectData
from datasets import register_dataset
from pathlib import Path
from .utils import save_cache
import logging

log = logging.getLogger(__name__)

class BaseRegressionDataset(data.Dataset):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.cache_input_features = None
        if "cache_input_features" in self.cfg:
            self.cache_input_features = self.cfg.cache_input_features

    def get_input_dim(self):
        item = self.__getitem__(0)
        return item["input"].shape[-1]

    def get_output_size(self):
        return 1 #single logit

    def __len__(self):
        return self.seeg_data.shape[0]

    def label2idx(self, label):
        return self.label2idx_dict[label]

@register_dataset(name="nounverb_regression")
class NounVerbFinetuning(BaseRegressionDataset):
    def __init__(self, cfg, task_cfg=None, preprocessor_cfg=None, cached_word_df=None, cached_seeg_data=None) -> None:
        super().__init__(cfg)

        if cached_word_df is not None and cached_seeg_data is not None:
            word_df = cached_word_df
            seeg_data = cached_seeg_data
        else:
            s = SubjectData(cfg)
            word_df = s.words
            seeg_data = s.neural_data

        assert len(cfg.electrodes) == 1
        assert seeg_data.shape[0] == 1
        seeg_data = seeg_data.squeeze(0)

        noun_verb_idxs = word_df.pos.isin(["NOUN","VERB"]).tolist()

        if 'position_subsample' in cfg: 
            subsample_idxs = (word_df[cfg.position_subsample])
            if cfg.position_subsample=="is_midset":
                subsample_idxs = self.equalize_by_exact_index(word_df, is_midset=True)
        else:
            subsample_idxs = self.equalize_by_exact_index(word_df, is_midset=True)
        noun_verb_idxs = np.array(subsample_idxs)*np.array(noun_verb_idxs)
        noun_verb_idxs = [bool(x) for x in noun_verb_idxs]

        poss = set(word_df[noun_verb_idxs].pos)
        self.word_df = word_df[noun_verb_idxs].reset_index(drop=True)#might still be used downstream
        self.seeg_data = seeg_data[noun_verb_idxs]

        label2idx_dict = {}
        for idx, pos in enumerate(poss):
            label2idx_dict[pos] = idx
        self.label2idx_dict = label2idx_dict
        self.idx2label_dict = {k:v for v,k in label2idx_dict.items()}
        self.label = [self.label2idx(x) for x in self.word_df.pos]

    def equalize_by_exact_index(self, word_df, is_midset=False):
        subsample_idxs = np.array([False]*len(word_df))
        assert all(word_df.index==list(range(len(word_df))))
        midset_words = word_df[subsample_idxs].index
        max_idx = int(max(word_df.idx_in_sentence))

        sentence_index_start = 0
        sentence_index_end = max_idx
        if is_midset:
            sentence_index_start = 1
            sentence_index_end = max_idx-1

        for i in range(sentence_index_start,sentence_index_end):
            #balance per index
            midset_noun_idxs = word_df[(word_df.idx_in_sentence==i) & (word_df.pos=="NOUN")].index
            midset_verb_idxs = word_df[(word_df.idx_in_sentence==i) & (word_df.pos=="VERB")].index
            min_num = min(len(midset_noun_idxs), len(midset_verb_idxs))
            midset_noun_idxs = midset_noun_idxs[:min_num]
            midset_verb_idxs = midset_verb_idxs[:min_num]
            subsample_idxs[midset_noun_idxs] = True
            subsample_idxs[midset_verb_idxs] = True

        return subsample_idxs

    def __getitem__(self, idx: int):
        #NOTE: remember not to load to cuda here
        input = self.seeg_data[idx]
        label = self.label[idx]
        return {
                "input": input,
                "label": label, 
               }

@register_dataset(name="word_onset_regression")
class WordOnsetRegression(BaseRegressionDataset):
    def __init__(self, cfg, task_cfg=None, preprocessor_cfg=None, cached_word_df=None, cached_seeg_data=None) -> None:
        super().__init__(cfg)

        if cached_word_df is not None and cached_seeg_data is not None:
            word_df = cached_word_df
            seeg_data = cached_seeg_data
        else:
            s = NonLinguisticSubjectData(cfg)
            word_df = s.labels
            seeg_data = s.neural_data
            import pdb; pdb.set_trace()

        assert len(cfg.electrodes) == 1
        assert seeg_data.shape[0] == 1
        seeg_data = seeg_data.squeeze(0)

        labels = set(word_df.linguistic_content)
        self.word_df = word_df.reset_index(drop=True)#might still be used downstream
        self.seeg_data = seeg_data

        label2idx_dict = {}
        for idx, l in enumerate(labels):
            label2idx_dict[l] = idx
        self.label2idx_dict = label2idx_dict
        self.idx2label_dict = {k:v for v,k in label2idx_dict.items()}
        self.label = [self.label2idx(x) for x in self.word_df.linguistic_content]

@register_dataset(name="sentence_onset_regression")
class SentenceOnsetRegression(BaseRegressionDataset):
    def __init__(self, cfg, task_cfg=None, preprocessor_cfg=None, cached_word_df=None, cached_seeg_data=None) -> None:
        super().__init__(cfg)

        if cached_word_df is not None and cached_seeg_data is not None:
            word_df = cached_word_df
            seeg_data = cached_seeg_data
        else:
            s = SentenceOnsetSubjectData(cfg)
            word_df = s.labels
            seeg_data = s.neural_data
            import pdb; pdb.set_trace()

        assert len(cfg.electrodes) == 1
        assert seeg_data.shape[0] == 1
        seeg_data = seeg_data.squeeze(0)

        labels = set(word_df.linguistic_content)
        self.word_df = word_df.reset_index(drop=True)#might still be used downstream
        self.seeg_data = seeg_data

        label2idx_dict = {}
        for idx, l in enumerate(labels):
            label2idx_dict[l] = idx
        self.label2idx_dict = label2idx_dict
        self.idx2label_dict = {k:v for v,k in label2idx_dict.items()}
        self.label = [self.label2idx(x) for x in self.word_df.linguistic_content]
