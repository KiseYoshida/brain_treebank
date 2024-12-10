import numpy as np
from omegaconf import DictConfig, OmegaConf
import hydra
import models
import pandas
import tasks
from glm_runner import GLMRunner
import logging
import os
from data.electrode_selection import get_all_electrodes
import json
from pathlib import Path
from datasets import build_dataset

log = logging.getLogger(__name__)

def run_subject_test(data_cfg, brain_runs, electrodes, cfg):
    data_cfg_copy = data_cfg.copy()
    for e in electrodes:
        data_cfg_copy.electrodes = [e]
        data_cfg_copy.brain_runs = brain_runs
        dataset = build_dataset(data_cfg_copy, task_cfg=cfg)#build glm_dataset
        runner = GLMRunner(cfg, dataset)
        res = runner.run(e)

        coef_df = res.coefs
        flat_df = flatten_df(coef_df)
        rsquared = pymer4.stats.rsquared(res.data.target, res.data.residuals, has_constant=True)
        pos_feature = (res.data.pos=="NOUN").astype(int)
        flat_df['rsquared'] = rsquared
        coef_df_lst.append(flat_df)
        if e_idxs < len(elec_idx_lst) - 2:
            all_elec_means.append(res.data.target.mean())

    print("all means", np.mean(all_elec_means))
    summary_df = pd.concat(coef_df_lst, axis=1)
    summary_df.columns = elec_label_lst

@hydra.main(config_path="conf")
def main(cfg: DictConfig) -> None:
    log.info(f"Run glms for all electrodes in all subjects")
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
            electrodes = get_all_electrodes(subj, data_root=data_cfg.data_root)
        trials = test_splits[subj]
        subject_test_results = run_subject_test(data_cfg, trials, electrodes, cfg)

if __name__ == "__main__":
    main()
