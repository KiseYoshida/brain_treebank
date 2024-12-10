from data.subject_data import SubjectData
import logging
import os
from omegaconf import DictConfig, OmegaConf
from data.electrode_selection import get_all_subjects_all_electrodes
import hydra
import json
from pathlib import Path
import numpy as np
from scipy import stats
from glob import glob
import pandas as pd

log = logging.getLogger(__name__)
FLOAT_FEATURES = ["delta_pitch", "word_length", "rms", "pitch", "max_vector_magnitude", "mean_pixel_brightness", "max_global_angle", "gpt2_surprisal", "delta_rms", "face_num", "max_global_magnitude", "max_vector_angle", "word_diff", "bin_head"] #NOTE that is_onset and bin_head are categorical variables, but for our purposes, can be treated as a float
CAT_FEATURES = ["idx_in_sentence", "pos"]
samp_rate = 2048

def get_subject_df(file_path):
    subj, target = file_path.split('/')[-3:-1]
    print(subj, target)
    df = pd.read_csv(file_path)
    df = standardize_df(df, subj)
    return df

def remove_intercept(df):
    res = df[~df.label.str.contains('Intercept')]
    return res

def standardize_df(df, subj):
    df = df.copy()
    
    df = df.loc[:,~df.columns.isin(['DC10', 'DC4', 'TRIG4'])]

    keys = df.keys()
    new_keys = list(keys).copy()
    new_keys[0] = 'label'
    new_keys = [f'{x}_{subj}' if x != 'label' else x for x in new_keys ]
    mapping = {k:v for k,v in zip(keys,new_keys)}
    df.rename(columns=mapping, inplace=True)
    
    df.columns = df.columns.str.replace("\#",'',regex=True)
    df.columns = df.columns.str.replace("\*",'',regex=True)
    def rename_pos(x):
        all_words = x.split('-')
        init = all_words[:-1]
        last = all_words[-1]
        if last in ["posVERB", "pos[T.VERB]"]:
            new_all_words = init + ["posVERB"]
            return '-'.join(new_all_words)
        return x
    df.label = df.label.apply(rename_pos)
    df = remove_intercept(df)
    df.loc[~df.label.str.startswith('Sig'),~df.columns.isin(['label'])] = df.loc[~df.label.str.startswith('Sig'),~df.columns.isin(['label'])].astype('float32')
    return df

def get_control_idxs(features, control):
    '''
        features -- dataframe of word features per word
        control -- the name of a feature, e.g. rms
        returns
            a list of indexs, each of which correspond with a subsample of the control conditions
    '''
    #restrict attention to high (low) rms and compare between targets
    if control in FLOAT_FEATURES:
        high = np.percentile(features[control], 75)
        low = np.percentile(features[control], 25)
        if control in ["bin_head", "is_onset"]:
            low, high = 0.4, 0.6
        if high <= low or np.isnan(high) or np.isnan(low):
            #if we make it in this branch, just sort by the control feature and separate into the top and bottom quarter
            sorted_idx = features[control].sort_values().index
            quarter = int(len(features)/4)
            control_high_idxs = sorted_idx[-quarter:]
            control_low_idxs = sorted_idx[:quarter]
        else:
            assert high > low 
            control_high_idxs = features[features[control] >= high].index
            control_low_idxs = features[features[control] <= low].index
        assert len(control_high_idxs) > 2 and len(control_low_idxs) > 2
        return [control_high_idxs, control_low_idxs]
    elif control == "idx_in_sentence":
        onset_idxs = features[features["is_onset"]==1].index
        offset_idxs = features[features["is_offset"]==1].index
        midset_idxs = features[(features["is_onset"]==0) & (features["is_offset"]==0)].index
        return [onset_idxs, midset_idxs, offset_idxs]
    elif control == "pos":
        noun_idxs = features[features["pos"]=="NOUN"].index
        verb_idxs = features[features["pos"]=="VERB"].index
        return [noun_idxs, verb_idxs]
    else:
        print("control not found")
        import pdb; pdb.set_trace()    

def ttest_target_within_control(control_idxs, pre_data, post_data):
    pre_data = pre_data[:,control_idxs,:].mean(axis=-1)
    post_data = post_data[:,control_idxs,:].mean(axis=-1)
    tstat, pval = stats.ttest_rel(pre_data, post_data, axis=-1)

    mean_diff = pre_data.mean() - post_data.mean()
    std = np.std(np.concatenate([pre_data, post_data], axis=1).flatten())
    cohens_d = mean_diff/std
    return pval, cohens_d

def equalize_target_among_control(all_control_idxs, data_arr):
    min_size = min([len(l) for l in all_control_idxs])
    truncated_idxs = [l[:min_size] for l in all_control_idxs]
    equalized_idxs = np.concatenate(truncated_idxs)
    return data_arr[:,equalized_idxs,:]

def ttest_target_equalize_control(all_control_idxs, pre_data, post_data):
    '''
        all_control_idxs is a list of lists. each member list is a list of indexes.
    '''
    pre_data_eq = equalize_target_among_control(all_control_idxs, pre_data)
    post_data_eq = equalize_target_among_control(all_control_idxs, post_data)
    pre_mean = pre_data_eq.mean(axis=-1)
    post_mean = post_data_eq.mean(axis=-1)
    tstat, pval = stats.ttest_ind(pre_mean, post_mean, axis=-1)

    mean_diff = pre_mean.mean() - post_mean.mean()
    std = np.std(np.concatenate([pre_mean, post_mean], axis=1).flatten())
    cohens_d = mean_diff/std
    assert pval.shape==(1,)
    return pval.item(), cohens_d

def uncontrolled_ttest(subj_data, cfg):
    '''
    returns ttest results for pre and post onset
    data_arr shape [n_electrodes, n_words, n_timesteps]
    '''
    min_pval = 1
    cohens = None
    for pre_onset_start in np.arange(cfg.exp.search_start, cfg.exp.search_end, cfg.exp.interval):
        pre_onset_start = float(pre_onset_start)
        cfg.exp.pre_onset_start = pre_onset_start
        cfg.exp.pre_onset_end = pre_onset_start + cfg.exp.interval
        cfg.exp.post_onset_start = pre_onset_start + cfg.exp.separation
        cfg.exp.post_onset_end = pre_onset_start + cfg.exp.separation + cfg.exp.interval
        pre_data, post_data = get_pre_post_data(subj_data.neural_data, cfg)

        assert pre_data.shape[1] == post_data.shape[1]
        assert len(pre_data.shape) == 3 and len(post_data.shape) == 3

        pre_data, post_data = pre_data.mean(axis=-1), post_data.mean(axis=-1)
        tstat, pval = stats.ttest_rel(pre_data, post_data, axis=-1)
        mean_diff = pre_data.mean() - post_data.mean()
        std = np.std(np.concatenate([post_data, pre_data], axis=1).flatten())
        d = mean_diff/std

        if pval < min_pval:
            min_pval = pval
            cohens = d

    return pval.item(), cohens


def run_ttest(subj_data, features, control, control_type, cfg):
    '''
    control - e.g. volume
    control_type -- equalize or sub_sample
    returns ttest results for high and low volume
    data_arr shape [n_electrodes, n_words, n_timesteps]
    '''
    pval = 0.001
    assert len(features) == features.index[-1]+1

    all_control_idxs = get_control_idxs(features, control)

    all_pvals, all_cohens = [], []
    if control_type=="sub_sample":
        for control_idxs in all_control_idxs:
            min_pval = 1
            cohens = None
            for pre_onset_start in np.arange(cfg.exp.search_start, cfg.exp.search_end, cfg.exp.interval):
                pre_onset_start = float(pre_onset_start)
                cfg.exp.pre_onset_start = pre_onset_start
                cfg.exp.pre_onset_end = pre_onset_start + cfg.exp.interval
                cfg.exp.post_onset_start = pre_onset_start + cfg.exp.separation
                cfg.exp.post_onset_end = pre_onset_start + cfg.exp.separation + cfg.exp.interval
                pre_data, post_data = get_pre_post_data(subj_data.neural_data, cfg)
                assert pre_data.shape[1] == post_data.shape[1]
                assert pre_data.shape[1] == features.shape[0]
                assert len(pre_data.shape) == 3 and len(post_data.shape) == 3
                pval, d = ttest_target_within_control(control_idxs, pre_data, post_data)
                if pval < min_pval:
                    min_pval = pval
                    cohens = d
            all_pvals.append(min_pval)
            all_cohens.append(cohens)
        all_pvals = np.concatenate(all_pvals)
        return all_pvals.tolist(), all_cohens
    elif control_type=="equalize":
        print("This analysis is not applicable for word onsets")
        import pdb; pdb.set_trace()
        #pval, d = ttest_target_equalize_control(all_control_idxs, pre_data, post_data)
        #return [pval], [d]
    else:
        print("not found")
        import pdb; pdb.set_trace()
    
def run_all_features(subj_data, words_df, sig_features, subj, elec, cfg, control_type="sub_sample"):
    all_results = {}
    other_features = sig_features

    for other_feature in other_features:
        pvals, cohens = run_ttest(subj_data, words_df, other_feature, control_type, cfg)
        all_results[other_feature] = {"pvals": pvals,
                                      "cohens": cohens}
    return all_results

def get_pre_post_data(data, cfg):
    pre_start = cfg.exp.pre_onset_start
    pre_end = cfg.exp.pre_onset_end
    post_start = cfg.exp.post_onset_start
    post_end = cfg.exp.post_onset_end

    data_delta = cfg.data.delta

    assert pre_start < pre_end and pre_end <= 0
    assert post_end > post_start and post_start >= 0

    onset_idx = int(-samp_rate*data_delta)
    pre_start_idx = onset_idx + int(samp_rate*pre_start)
    pre_end_idx = onset_idx + int(samp_rate*pre_end)

    post_start_idx = onset_idx + int(samp_rate*post_start)
    post_end_idx = onset_idx + int(samp_rate*post_end)

    assert len(data.shape)==3
    assert post_end_idx <= data.shape[-1]
    assert pre_start_idx >= 0

    pre_data = data[:,:,pre_start_idx:pre_end_idx]
    post_data = data[:,:,post_start_idx:post_end_idx]
    return pre_data, post_data
           
@hydra.main(config_path="../conf", version_base=None)
def main(cfg: DictConfig) -> None:
    log.info(f"Run word onset testing for all electrodes in all test_subjects")
    log.info(OmegaConf.to_yaml(cfg, resolve=True))
    out_dir = cfg.exp.get("out_dir", None)
    if not out_dir:
        out_dir = os.getcwd()
    else:
        Path(out_dir).mkdir(exist_ok=True, parents=True)
    log.info(f'Working directory {out_dir}')

    all_elecs = get_all_subjects_all_electrodes(data_root=cfg.data.data_root)

    test_split_path = cfg.test.test_split_path 
    with open(test_split_path, "r") as f:
        test_splits = json.load(f)

    data_cfg = cfg.data
    all_test_results = {}

    for elec_subj in all_elecs:
        subj,elec = elec_subj
        log.info(f"Subject {subj}")
        data_cfg.subject = subj

        data_cfg.electrodes = [elec]
        data_cfg.brain_runs = test_splits[subj]

        all_features = ['pos', 'bin_head', 'gpt2_surprisal', 'word_diff', 'word_length', 'rms', 'pitch', 'delta_rms', 'delta_pitch', 'mean_pixel_brightness', 'max_global_magnitude', 'max_global_angle', 'max_vector_magnitude', 'max_vector_angle', 'face_num', 'idx_in_sentence']
        assert len(set(FLOAT_FEATURES).intersection(set(all_features))) == len(FLOAT_FEATURES)

        subj_data = SubjectData(data_cfg)
        words_df = subj_data.words.reset_index(drop=True)
        elec_results = run_all_features(subj_data, words_df, all_features, subj, elec, cfg, cfg.exp.control_type)
        pval, cohens = uncontrolled_ttest(subj_data, cfg)
        elec_results["uncontrolled"] = {"pvals":[pval], "cohens":[cohens]}
        all_test_results[f"{elec}_{subj}"] = elec_results
    results_path = os.path.join(out_dir, f'all_results.json')
    with open(results_path, "w") as f:
        json.dump(all_test_results, f)
    log.info(f'Working directory {out_dir}')

if __name__ == "__main__":
    main()

