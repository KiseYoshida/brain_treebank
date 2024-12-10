from data.subject_data import SubjectData
import numpy as np
from omegaconf import DictConfig, OmegaConf
import hydra
import models
import tasks
from runner import Runner
from regression_runner import RegressionRunner, RegressionRunnerTestSet
from data.speech_nonspeech_subject_data import NonLinguisticSubjectData, SentenceOnsetSubjectData
import logging
import os
from data.electrode_selection import get_all_electrodes
import json
from pathlib import Path
from tqdm import tqdm as tqdm
from datasets import build_dataset
import random

log = logging.getLogger(__name__)
SR = 2048

def get_crossval_indices(dataset,  n_splits=10):
    #get set of train and val datasets
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    split_indices = np.array_split(indices, int(n_splits*2))

    all_train_indices = []
    all_val_indices = []
    all_test_indices = []
    for i in range(n_splits):
        val_indices = split_indices[2*i]
        test_indices = split_indices[2*i+1]
        train_indices = np.concatenate(split_indices[:2*i] + split_indices[2*i+2:])
        all_train_indices.append(val_indices)
        all_test_indices.append(test_indices)
        all_val_indices.append(train_indices)
    return all_train_indices, all_val_indices, all_test_indices

def nn_runner_cross_validate(cfg, data_cfg, cached_seeg_data, cached_word_df):
    cfg.data = data_cfg#Is this going to mess up cfg.data permanently?
    all_dataset = build_dataset(data_cfg, task_cfg=cfg.task, cached_seeg_data=cached_seeg_data, cached_word_df=cached_word_df)
    all_train_indices, all_val_indices, all_test_indices = get_crossval_indices(all_dataset, n_splits=10)#for loop over crossval indices

    results = []
    for train_indices, val_indices, test_indices in zip(all_train_indices, all_val_indices, all_test_indices):
        task = tasks.setup_task(cfg.task)
        task.load_datasets(cfg.data, cached_seeg_data=cached_seeg_data, cached_word_df=cached_word_df, train_indices=train_indices, val_indices=val_indices, test_indices=test_indices) #maybe here we can set the index?
        model = task.build_model(cfg.model)
        criterion = task.build_criterion(cfg.criterion)

        runner = Runner(cfg.exp.runner, task, model, criterion)
        best_model = runner.train()
        test_results = runner.test(best_model)
        results.append(test_results["roc_auc"])
    return results

def run_electrode_test(data_cfg, cfg, cached_seeg_data, cached_word_df, count=0):
    #TODO cache the subject data
    cfg.data = data_cfg#Is this going to mess up cfg.data permanently?
    repeat_results = []
    task = tasks.setup_task(cfg.task)

    if cfg.exp.runner.name == "regression_runner":
        task.load_datasets(cfg.data, cached_seeg_data=cached_seeg_data, cached_word_df=cached_word_df)
        model = task.build_model(cfg.model)
        criterion = task.build_criterion(cfg.criterion)
        runner = RegressionRunner(cfg.exp.runner, task, model, criterion)
        results = runner.cross_validate()
        results = results.tolist()
    elif cfg.exp.runner.name == "regression_runner_test_set":
        task.load_datasets(cfg.data, cached_seeg_data=cached_seeg_data, cached_word_df=cached_word_df)
        model = task.build_model(cfg.model)
        criterion = task.build_criterion(cfg.criterion)
        runner = RegressionRunnerTestSet(cfg.exp.runner, task, model, criterion)
        results = runner.split_and_train_and_test(count=count)
    elif cfg.exp.runner.name == "nn_runner":
        #let's create a runner per each train cross validate split
        #and then let's get the  test results
        results = nn_runner_cross_validate(cfg, data_cfg, cached_seeg_data, cached_word_df)
    else:
        print("no runner")

    return results

def run_subject_test(data_cfg, brain_runs, electrodes, cfg):
    data_cfg_copy = data_cfg.copy()
    #cache_path = None
    #if "cache_input_features" in data_cfg_copy:
    #    cache_path = data_cfg_copy.cache_input_features

    windows_start = cfg.task.windows_start
    windows_end = cfg.task.windows_end
    all_windows_duration = windows_end - windows_start
    window_duration = cfg.task.window_duration
    window_step = cfg.task.window_step
    assert windows_end > windows_start 
    assert (windows_end - windows_start) > window_duration
    window_starts = np.arange(windows_start, windows_end, window_step)
    sr = 2048
    idx_starts = [int((w - windows_start)*sr) for w in window_starts]
    idx_duration = int(window_duration*2048)

    subject_test_results = {}
    
    for e in electrodes:
        data_cfg_copy = data_cfg.copy()
        data_cfg_copy.duration = all_windows_duration
        data_cfg_copy.delta = windows_start
        data_cfg_copy.electrodes = [e]
        data_cfg_copy.brain_runs = brain_runs

        if cfg.data.name == "nounverb_regression":
            s = SubjectData(data_cfg_copy)
            cached_word_df = s.words
            all_data = s.neural_data
        elif cfg.data.name == "word_onset_regression":
            s = NonLinguisticSubjectData(data_cfg_copy)
            cached_word_df = s.labels
            all_data = s.neural_data
        elif cfg.data.name == "sentence_onset_regression":
            s = SentenceOnsetSubjectData(data_cfg_copy)
            cached_word_df = s.labels
            all_data = s.neural_data
        else:
            print("dataset not supported")
            import pdb; pdb.set_trace()

        #import pdb; pdb.set_trace()
        data_cfg_copy.duration = "???"
        data_cfg_copy.delta = "???"

        elec_test_results = {}
        log.info(f'running across all time windows {e}')
        for i,start in tqdm(enumerate(idx_starts)):
            #assert start+idx_duration <= all_data.shape[-1]
            cached_seeg_data = all_data[:,:,start:start+idx_duration]
            elec_test_results[window_starts[i]] = run_electrode_test(data_cfg_copy, cfg, cached_seeg_data, cached_word_df, count=i)
        subject_test_results[e] = elec_test_results
    return subject_test_results

def write_summary(all_test_results, out_path):
    out_json = os.path.join(out_path, "all_test_results.json")
    with open(out_json, "w") as f:
        json.dump(all_test_results, f)

    out_json = os.path.join(out_path, "summary.json")
    all_rocs = []
    for s in all_test_results:
        for e in all_test_results[s]:
            for t in all_test_results[s][e]:
                all_rocs += all_test_results[s][e][t]

    summary_results = {"avg_roc_auc": np.mean(all_rocs), "std_roc_auc": np.std(all_rocs)}
    with open(out_json, "w") as f:
        json.dump(summary_results, f)

    log.info(f"Wrote test results to {out_path}")

@hydra.main(config_path="conf")
def main(cfg: DictConfig) -> None:
    log.info(f"Run testing for all electrodes in all test_subjects across all times")
    log.info(OmegaConf.to_yaml(cfg, resolve=True))
    out_dir = os.getcwd()
    log.info(f'Working directory {os.getcwd()}')
    if "out_dir" in cfg.test:
        out_dir = cfg.test.out_dir
    log.info(f'Output directory {out_dir}')

    test_split_path = cfg.test.test_split_path 
    with open(test_split_path, "r") as f:
        test_splits = json.load(f)

    test_electrodes = None #For the topk. Omit this argument if you want everything
    if "test_electrodes_path" in cfg.test and cfg.test.test_electrodes_path != "None": #very hacky
        test_electrodes_path = cfg.test.test_electrodes_path 
        test_electrodes_path = os.path.join(test_electrodes_path, cfg.data.name)
        test_electrodes_path = os.path.join(test_electrodes_path, "linear_results.json")
        with open(test_electrodes_path, "r") as f:
            test_electrodes = json.load(f)

    data_cfg = cfg.data
    all_test_results = {}
    for subj in test_splits:
        subj_test_results = {}
        Path(out_dir).mkdir(exist_ok=True, parents=True)
        out_path = os.path.join(out_dir, "all_test_results")
        log.info(f"Subject {subj}")
        data_cfg.subject = subj
        if test_electrodes is not None:
            if subj not in test_electrodes:
                continue
            electrodes = test_electrodes[subj]
        else:
            electrodes = get_all_electrodes(subj, data_root=cfg.data.data_root)
        subject_test_results = run_subject_test(data_cfg, test_splits[subj], electrodes, cfg)
        all_test_results[subj] = subject_test_results
        subj_test_results[subj] = subject_test_results

        out_json_path = os.path.join(out_path, subj)
        Path(out_json_path).mkdir(exist_ok=True, parents=True)
        out_json = os.path.join(out_json_path, "subj_test_results.json")
        with open(out_json, "w") as f:
            json.dump(subj_test_results, f)
        log.info(f"Wrote test results to {out_json}")
    write_summary(all_test_results, out_path)

if __name__ == "__main__":
    main()

