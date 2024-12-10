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

samp_rate = 2048

log = logging.getLogger(__name__)
FLOAT_FEATURES = ["delta_magnitude", "delta_pitch", "word_length", "rms", "pitch", "max_vector_magnitude", "mean_pixel_brightness", "max_global_angle", "gpt2_surprisal", "delta_rms", "face_num", "is_onset", "max_global_magnitude", "max_vector_angle", "max_mean_magnitude", "word_diff", "bin_head"] #NOTE that is_onset and bin_head are categorical variables, but for our purposes, can be treated as a float
CAT_FEATURES = ["idx_in_sentence", "pos"]


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

def get_all_results(results_dir):
    files = glob(f'{results_dir}/*/*/*.csv')
    files = [f for f in files if 'amp' in f]
    all_dfs = []
    for file_path in files:
        df = get_subject_df(file_path)
        df = df.set_index("label")
#         print(df.index.tolist())
        if 'Estimate-phoneme_num' in df.index.tolist():
            print(file_path)
        all_dfs.append(df)

    all_results = pd.concat(all_dfs,axis=1)
    return all_results

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

def test_population_means_target_within_control(control_idxs, data_arr, all_targets):
    target_controls = [control_idxs.intersection(t_idxs) for t_idxs in all_targets]
    #if min([len(t) for t in target_controls]) <= 0:
    #    import pdb; pdb.set_trace()
    assert min([len(t) for t in target_controls]) > 0
    target_data = [data_arr[:,t_idxs,:].mean(axis=-1) for t_idxs in target_controls]
    assert all([t.shape[0]==1 for t in target_data])
    target_data_lists = [t.flatten().tolist() for t in target_data]
    tstat, pval = stats.f_oneway(*target_data_lists)

    #mean_diff = target_1.mean() - target_2.mean()
    #std = np.std(np.concatenate([target_1, target_2], axis=1).flatten())
    #cohens_d = mean_diff/std
    return pval, 0#, cohens_d

def equalize_target_among_control(all_control_idxs, target_idxs):
    controlled_target = [l.intersection(target_idxs) for l in all_control_idxs]
    min_size = min([len(l) for l in controlled_target])
    truncated_idxs = [l[:min_size] for l in controlled_target]
    equalized_idxs = np.concatenate(truncated_idxs)
    return equalized_idxs

def test_population_means_equalize_target(all_control_idxs, data_arr, target_idxs):
    '''
        all_control_idxs is a list of lists. each member list is a list of indexes.
    '''
    target_controls = [equalize_target_among_control(all_control_idxs, t_idxs) for t_idxs in target_idxs]
    assert min([len(t) for t in target_controls]) > 0
    target_data = [data_arr[:,t_idxs,:].mean(axis=-1) for t_idxs in target_controls]
    assert all([t.shape[0]==1 for t in target_data])
    target_data_lists = [t.flatten().tolist() for t in target_data]
    tstat, pval = stats.f_oneway(*target_data_lists)

    #mean_diff = target_1.mean() - target_2.mean()
    #std = np.std(np.concatenate([target_1, target_2], axis=1).flatten())
    #cohens_d = mean_diff/std
    return pval, 0#, cohens_d

def run_comparison(data_arr, features, target, control, control_type, cfg):
    '''
    target - usually part-of-speech. The variable we are interested in. Are verbs and nouns different?
    control - e.g. volume
    control_type -- equalize or sub_sample
    returns ttest results for high and low volume
    data_arr shape [n_electrodes, n_words, n_timesteps]
    '''
    pval = 0.001
    assert data_arr.shape[1] == features.shape[0]
    assert len(data_arr.shape) == 3
    assert len(features) == features.index[-1]+1

    all_target_idxs = get_control_idxs(features, target)
    all_control_idxs = get_control_idxs(features, control)

    all_pvals, all_cohens = [], []
    if control_type=="sub_sample":
        for control_idxs in all_control_idxs:
            min_pval = 1
            cohens = None
            data_delta = cfg.data.delta
            for interval_start in np.arange(cfg.exp.search_start, cfg.exp.search_end, cfg.exp.interval):
                #print(interval_start)
                assert interval_start > data_delta
                assert interval_start + cfg.exp.interval < data_delta + cfg.data.duration

                interval_start_idx = int((interval_start-data_delta)*samp_rate)
                interval_end_idx = interval_start_idx + int(samp_rate*cfg.exp.interval)

                truncated_data_arr = data_arr[:,:,interval_start_idx:interval_end_idx]
                #print(interval_start_idx, interval_end_idx, data_arr.shape)
                pval, d = test_population_means_target_within_control(control_idxs, truncated_data_arr, all_target_idxs)
                if pval < min_pval:
                    min_pval = pval
                    cohens = d
            #if control=="word_diff":
            #    import pdb; pdb.set_trace()
            all_pvals.append(min_pval)
            all_cohens.append(cohens)
        #if control=="word_diff":
        #    import pdb; pdb.set_trace()

        return all_pvals, all_cohens
    elif control_type=="equalize":
        pval, d = test_population_means_equalize_target(all_control_idxs, data_arr, all_target_idxs)
        return [pval], [d]
    else:
        print("not found")
        import pdb; pdb.set_trace()
    
def run_all_features(neural_data, words_df, sig_features, subj, elec, target, cfg, control_type="sub_sample"):
    all_results = {}
    other_features = sig_features
    if target=="sentence_pos":
        if "is_onset" in other_features:
            other_features.remove("is_onset")#this interferes with idx_in_sentence
        target = "idx_in_sentence"
            
    if target in sig_features:
        other_features.remove(target)

    for other_feature in other_features:
        pvals, cohens = run_comparison(neural_data, words_df, target, other_feature, control_type, cfg)
        all_results[other_feature] = {"pvals": pvals,
                                      "cohens": cohens}
    return all_results
           
@hydra.main(config_path="../conf")
def main(cfg: DictConfig) -> None:
    log.info(f"Run testing for all electrodes in all test_subjects")
    log.info(OmegaConf.to_yaml(cfg, resolve=True))
    assert cfg.exp.control_type in ["sub_sample", "equalize"]
    '''
        sub_sample -- find rms high and rms low. For each domain, compare nouns vs. verbs
        equalize -- find rms high and rms low. equalize each for nouns and equalize each for verbs. compare n vs v.
    '''
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

    target = "sentence_pos"
    for subj_elec in all_elecs:

        subj,elec = subj_elec
        subj,elec = ("sub_4","LT2aA1")#TODO
        log.info(f"Subject {subj}")
        data_cfg.subject = subj

        data_cfg.electrodes = [elec]
        data_cfg.brain_runs = test_splits[subj]

        all_features = ['pos', 'bin_head', 'gpt2_surprisal', 'word_diff', 'word_length', 'rms', 'pitch', 'delta_rms', 'delta_pitch', 'mean_pixel_brightness', 'max_global_magnitude', 'max_global_angle', 'max_vector_magnitude', 'max_vector_angle', 'face_num', 'idx_in_sentence']

        data_cfg
        subj_data = SubjectData(data_cfg)
        words_df = subj_data.words.reset_index(drop=True)
        elec_results = run_all_features(subj_data.neural_data, words_df, all_features, subj, elec, target, cfg, cfg.exp.control_type)

        all_test_results[subj_elec] = elec_results

    results_path = os.path.join(out_dir, f'all_results.json')
    with open(results_path, "w") as f:
        json.dump(all_test_results, f)
    log.info(f'Working directory {out_dir}')

if __name__ == "__main__":
    main()
