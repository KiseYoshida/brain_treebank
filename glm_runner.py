import pandas as pd
import os
from pymer4.models import Lmer, Lm

class BaseLinearPredictor:
    samp_rate = 2048

    pos_col = 'pos'
    diff_col = 'word_diff'

    def __init__(self, args, data_loader, pred_lst):
        self.duration = args.data.duration
        self.delta = args.data.delta
        self.reload = True #args.reload #TODO hardcode
        #self.anno_source = args.anno_source
        #self.time_back = args.time_back
        #self.k_back = args.k_back
        self.pred_lst = pred_lst
        self.data_loader = data_loader
        #self.unit_variance = args.unit_variance
        self.unit_variance = True #TODO hardcode
        #self.sentence_position = args.sentence_position
        self.sentence_position = "all" #TODO hardcode
        self.X_dfs, self.samp_idxs = self.data_loader.get_all_predictors_dataframe(self.pred_lst)
        #self.permute_targets = args.permute_targets
        self.permute_targets = False #TODO hardcode
        self.verbose = False #NOTE hardcode
        #self.clear_cache = args.clear_cache
        self.clear_cache = True #TODO hardcode
        self.out_dir = args.test.out_dir

    @staticmethod
    def get_model_str(col_lst, y_str):
        mod_str = y_str + ' ~ '
        col_lst.remove(y_str)
        for col_str in col_lst:
            mod_str += col_str + ' + '
        mod_str = mod_str[:-3]
        return mod_str

    def get_r_model_str(self, col_lst, y_str):
        return self.get_model_str(col_lst, y_str)[:-len('trial')] + '(1|trial)'

    def remove_prev_features_by_time(self, df):
        tmp_df = df.copy(deep=True)
        if self.k_back > 0:
            prev_1_cols = [c for c in tmp_df.columns if 'prev_1_' in c]
            tmp_df.loc[tmp_df[self.diff_col] > self.time_back, prev_1_cols] = np.nan
            for i in range(1, self.k_back):
                tmp_df[[c for c in tmp_df.columns if 'prev_{}_'.format(i+1) in c]] = \
                    tmp_df[[c for c in tmp_df.columns if 'prev_{}_'.format(i) in c]].shift(i+1)
        return tmp_df.dropna()

    #def get_predictors_dataframe(self, pred_lst):
    #    pred_df = self.data_loader.get_predictors_dataframe()
    #    pred_df = pred_df.copy() #To avoid warnings about setting on a copy of a slice
    #    offset_list = pred_df.is_onset.shift(-1).to_list()
    #    offset_list[-1] = 1
    #    pred_df["is_offset"] = offset_list
    #    if self.sentence_position=="off":
    #        pred_df = pred_df[pred_df["is_offset"]==1]
    #    elif self.sentence_position=="on":
    #        pred_df = pred_df[pred_df["is_onset"]==1]
    #    elif self.sentence_position=="mid":
    #        pred_df = pred_df[(pred_df["is_onset"]!=1) & (pred_df["is_offset"]!=1)]
    #    else:
    #        assert self.sentence_position=="all"
    #    pred_lst = np.array(pred_lst)
    #    pos_feature_idx = np.where(['pos-' in p for p in pred_lst])[0]
    #    if len(pos_feature_idx) > 0:
    #        pos_feature_lst = np.array([pred_lst[i].split('-')[1] for i in pos_feature_idx])
    #        pred_lst = np.delete(pred_lst, pos_feature_idx)
    #        pred_df = pred_df.loc[pred_df[self.pos_col].str.lower().isin(pos_feature_lst)]
    #        pred_df = self.remove_prev_features_by_time(pred_df)
    #        assert not pred_df[pred_lst].mean().isnull().any()
    #        #X_df = (pred_df[pred_lst] - np.nanmean(pred_df[pred_lst])).dropna()
    #        X_df = (pred_df[pred_lst] - pred_df[pred_lst].mean()).dropna()
    #        if self.unit_variance:
    #            X_df = X_df/(X_df.std() + EPSILON)
    #        X_df[self.pos_col] = pred_df[self.pos_col]
    #        #for i in range(self.k_back):
    #        #    X_df['prev_{}_'.format(i+1) + self.pos_col] = pred_df['prev_{}_'.format(i+1) + self.pos_col]
    #    else:
    #        pred_df = self.remove_prev_features_by_time(pred_df)
    #        X_df = (pred_df[pred_lst] - np.nanmean(pred_df[pred_lst])).dropna()
    #    samp_idxs = X_df.index
    #    X_df = X_df.reset_index(drop=True)
    #    #self.full_event_df = pred_df.loc[self.samp_idxs].reset_index(drop=True)
    #    return X_df, samp_idxs

class GLMPredictorR(BaseLinearPredictor):
    def __init__(self, args, data_loader, pred_lst):
        super(GLMPredictorR, self).__init__(args, data_loader, pred_lst)

    def run(self, elec, target_type, peak_thresh=0):
        self.X_dfs
        for i,X_df in enumerate(self.X_dfs):
            X_df['trial'] = i
        X_df = pd.concat(self.X_dfs)
        #if self.permute_targets:
        #    random.shuffle(target)
        #    X_df.target = target

        model_str = self.get_r_model_str(X_df.columns.tolist(), 'target')
        model = Lmer(model_str, data=X_df)

        if self.verbose:
            print('\n', '#' * 35, self.data_loader.get_channel_label_by_index(elec), '#' * 35)
        model.fit(verbose=self.verbose, summarize=self.verbose, no_warnings=(not self.verbose))
        import pdb; pdb.set_trace()
        return model

class GLMRunner():
    def __init__(self, cfg, dataset):
        self.cfg = cfg
        self.exp_dir = os.getcwd()

        self.r_predictor = GLMPredictorR(cfg, dataset, dataset.pred_lst)#NOTE: eventually make this LME as well

    def run(self, elec):
        model = self.r_predictor.run(elec, "integrate")
        return model
