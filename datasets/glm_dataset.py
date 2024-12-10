from data.glm_subject_data import GLMSubjectData
from datasets import register_dataset
import numpy as np


EPSILON = 10**-9

@register_dataset("glm_dataset")
class GLMDataset():
    def __init__(self, cfg, task_cfg=None) -> None:

        super().__init__()
        data_cfg = cfg
        data_cfg_copy = data_cfg.copy()
        self.cfg = data_cfg_copy
        self.pred_lst = self.get_pred_lst(data_cfg.predictors_list_path)
        #import pdb; pdb.set_trace()
        s = GLMSubjectData(data_cfg_copy)
        self.subj_data = s
        self.steps_back = 0 #this is done to match the original data_loader code. Consider removing
        self.X_dfs, self.idxs = self.get_all_predictors_dataframe(self.pred_lst)

    def get_pred_lst(self, pred_lst_path):
        with open(pred_lst_path, "r") as f:
            lines = f.readlines()
            lines = [l.strip() for l in lines]
        return lines

    def get_all_predictors_dataframe(self, pred_lst):
        pred_dfs = self.subj_data.words
        neural_data = self.subj_data.neural_data
        X_dfs, idxs = zip(*[self.get_predictors_dataframe(pred_lst, pred_df, neural_arr) for (pred_df, neural_arr) in zip(pred_dfs, neural_data)])
        return X_dfs, idxs

    def integrate_voltage(self, neural_arr, idxs):
        #neural_data = self.subj_data.neural_data
        voltage_arr = neural_arr[0][idxs].mean(axis=1)
        return voltage_arr

    def get_target(self, target, neural_arr, idxs):
        if target == 'integrate':
            target = self.integrate_voltage(neural_arr, idxs)
        else:
            raise Exception('Target named \'{}\' isn\'t implemanted'.format(target))
        return target

    def get_predictors_dataframe(self, pred_lst, pred_df, neural_arr):
        pred_df = pred_df.copy() #To avoid warnings about setting on a copy of a slice
        #import pdb; pdb.set_trace()
        pred_df = pred_df.iloc[self.steps_back:]
        pred_lst = np.array(pred_lst)
        pos_feature_idx = np.where(['pos-' in p for p in pred_lst])[0]
        if len(pos_feature_idx) > 0:
            pos_feature_lst = np.array([pred_lst[i].split('-')[1] for i in pos_feature_idx])
            pred_lst = np.delete(pred_lst, pos_feature_idx)
            pred_df = pred_df.loc[pred_df["pos"].str.lower().isin(pos_feature_lst)]
            assert not any(['prev' in p for p in pred_lst])
            assert not pred_df[pred_lst].mean().isnull().any()
            pred_df = pred_df.dropna()
            X_df = (pred_df[pred_lst] - pred_df[pred_lst].mean()).dropna()

            unit_variance = self.cfg.get("unit_variance", False)
            if unit_variance:
                X_df = X_df/(X_df.std() + EPSILON)
            X_df["pos"] = pred_df["pos"]
        else:
            print("POS must be present")
            import pdb; pdb.set_trace()
            
        samp_idxs = X_df.index
        X_df = X_df.reset_index(drop=True)
        #self.full_event_df = pred_df.loc[self.samp_idxs].reset_index(drop=True)
        #import pdb; pdb.set_trace()
        #TODO get targets as well
        #import pdb; pdb.set_trace()
        X_df["target"] = self.get_target("integrate", neural_arr, samp_idxs)
        return X_df, samp_idxs

