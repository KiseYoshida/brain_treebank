# Brain Treebank
This repo contains the code for [Brain Treebank: Large-scale intracranial recordings
from naturalistic language stimuli](https://arxiv.org/pdf/2411.08343)

# Installing
## Data
Download data from https://braintreebank.dev/

The file organization should be:
```
braintreebank_data
  |_localization
  |_subject_timings
  |_subject_metadata
  |_electrode_labels
  |_transcripts
  |_all_subject_data
	  |_sub_*_trial*.h5
```
## Code environment
```
conda create -n "treebank" python=3.10
pip install -r requirements.txt
pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git  #not needed for non-deep NN experiments
conda install -c ejolly -c conda-forge -c defaults pymer4 #https://eshinjolly.com/pymer4/installation.html
```
# Analyses
## GLM analysis
This command runs the GLM analysis, as seen in our Figure 1, to determine the relative contribution of co-occurring features
```
python3 run_glms.py +exp=glm ++exp.runner.num_workers=0 +data=glm_template +model=glm_model +task=glm_task ++data.duration=0.5 ++data.delta=0 \
++data.interval_duration=1.0 +test=held_out_subjects \
++test.test_split_path=./data/all_trials.json ++test.test_electrodes_path="None"\
++test.out_dir="./glm_outputs" \
++data.name=glm_dataset ++data.predictors_list_path=./features_list.txt \
++data.unit_variance=True ++data.data_root=../braintreebank_data/ \
++data.raw_brain_data_dir=../braintreebank_data ++data.movie_transcripts_dir=../braintreebank_data/transcripts
```
Relevant arguments:
- `data.raw_brain_data_dir` and `data.data_root` have to point to `braintreebank_data` (see above)
- `data.movie_transcripts_dir` has to point to `braintreebank_data/transcripts` which contains the pre-computed movie features.
- `test.out_dir` is the location you want the results to be written to

## Word responsiveness
This commands runs the t-tests on the post-hoc stratified data, as seen in Figure 1:
```
python3 -m stats.word_onset_tests +data=tstat_word_onset +exp=word_onset ++data.duration=2.0 ++data.interval_duration=2.0 +test=all_trials ++data.delta=-1.0 \
++data.movie_transcripts_dir="../braintreebank_data/transcripts" \
++exp.control_type="sub_sample" \
++exp.out_dir="./word_onset_tstat_results" \
++data.data_root=/storage/czw/braintreebank_data ++test.test_split_path=./data/all_trials.json  \
++data.raw_brain_data_dir=../braintreebank_data/
```
Relevant arguments:
- `exp.out_dir` is where you want the results to be written.

## Modulation by sentence position
As above, this runs f-tests to determine if activity is modulated by a word's position in the sentence
```
python3 -m stats.sentence_position_tests +data=finetuning_template +exp=sentence_position \
++data.duration=3.0 ++data.interval_duration=1.0 \
++data.name="sentence_position_finetuning" +test=all_trials ++data.delta=-0.6 \
++data.movie_transcripts_dir="../braintreebank_data/transcripts" \
++exp.control_type="sub_sample" ++exp.out_dir="./sentence_tstat_results"  
++data.data_root=../braintreebank_data/ hydra.job.chdir=False \
++test.test_split_path=./data/all_trials.json ++data.raw_brain_data_dir=../braintreebank_data/
```
## Linear decoding
To run the linear decoding seen in our Figure 4:
```
python3 run_time_tests.py +exp=timed_tests ++exp.runner.num_workers=0 \
+data=decoding_base +model=classification_regression_model \
++exp.runner.name="regression_runner_test_set" +task=timed_tests +criterion=empty_criterion \
++data.interval_duration=3.0 ++data.name="sentence_onset_regression" +test=held_out_subjects \
++test.test_split_path=./data/all_trials.json ++test.test_electrodes_path="None" \
++data.movie_transcripts_dir="./braintreebank_data/transcripts" \
++task.windows_start=-1 ++task.windows_end=1 ++task.window_duration=0.250 ++task.window_step=0.1 \
++test.out_dir="./sentence_onset_linear_decoding_outputs/" \
++data.saved_data_split=./saved_data_splits ++data.cached_transcript_aligns=./saved_data_aligns \
++data.data_root=../braintreebank_data/ ++data.raw_brain_data_dir=../braintreebank_data/
```
Relevant arguments:
- `task.window_duration` is how much input (in seconds) each sliding window should contain
- `window_step` is how much the sliding window should be shifted

## Check alignment
```
python3 verify_frame_alignment.py cars-2 <movie_dir_path>/cars-2.mp4
```
