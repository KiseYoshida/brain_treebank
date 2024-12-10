from .utils import stem_electrode_name
import os
import json
import h5py
from glob import glob as glob

def get_all_laplacian_electrodes(elec_list):
    stems = [stem_electrode_name(e) for e in elec_list]
    def has_nbrs(stem, stems):
        (x,y) = stem
        return ((x,y+1) in stems) and ((x,y-1) in stems)
    laplacian_stems = [x for x in stems if has_nbrs(x, stems)]
    electrodes = [f'{x}{y}' for (x,y) in laplacian_stems]
    return electrodes

def get_all_electrodes(subject, data_root=None):
    '''
        returns list of electrodes in this subject and trial
        NOTE: the order of these labels is important. Their position corresponds with a row in data.h5
    '''
    electrode_labels_file = glob(os.path.join(data_root, "electrode_labels", subject, "electrode_labels.json"))
    assert len(electrode_labels_file)==1
    electrode_labels_file = electrode_labels_file[0]
    with open(electrode_labels_file, "r") as f:
        electrode_labels = json.load(f)
    strip_string = lambda x: x.replace("*","").replace("#","").replace("_","")
    electrode_labels = [strip_string(e) for e in electrode_labels]
    return electrode_labels

def clean_electrodes(subject, electrodes):
    corrupted_electrodes_path = "data/corrupted_elec.json"
    with open(corrupted_electrodes_path, "r") as f:
        corrupted_elecs = json.load(f)
    corrupt = corrupted_elecs[subject]
    return list(set(electrodes).difference(corrupt))

def get_clean_electrodes(subject):
    electrodes = get_all_electrodes(subject)
    electrodes = clean_electrodes(subject, electrodes)
    return electrodes

def get_clean_laplacian_electrodes(subject):
    electrodes = get_all_electrodes(subject)
    electrodes = clean_electrodes(subject, electrodes)
    laplacian_electrodes = get_all_laplacian_electrodes(electrodes)
    return laplacian_electrodes

def get_all_subjects_all_electrodes(data_root=None):
    all_electrodes = []
    with open("data/all_trials.json", "r") as f:
        subjects = json.load(f)
    for subject in subjects:
        electrodes = get_all_electrodes(subject, data_root=data_root)
        all_electrodes += [(subject, e) for e in electrodes]
    all_electrodes = [x for x in all_electrodes if ('DC' not in x[1] and 'TRIG' not in x[1])]
    return all_electrodes

def main():
    with open("data/pretrain_split_trials.json", "r") as f:
        subjects = json.load(f)
    all_electrodes = []
    for subject in subjects:
        electrodes = get_clean_laplacian_electrodes(subject)
        print(subject, len(electrodes))
        all_electrodes += [(subject, e) for e in electrodes]
    print(len(all_electrodes))

if __name__=="__main__":
    main()
