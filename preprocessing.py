import scipy.io
import numpy as np
import glob
import os
import cv2
import matplotlib.pyplot as plt
import mne

from sklearn.model_selection import train_test_split
from scipy.interpolate import griddata
from scipy.signal import welch


electrode_coords_10_20 = {
    'Fz': (0.0, 0.5), 'Cz': (0.0, 0.0), 'Pz': (0.0, -0.5), 'C3': (-0.3, 0.0),
    'T3': (-0.8, 0.0), 
    'C4': (0.3, 0.0),
    'T4': (0.8, 0.0), 
    'Fp1': (-0.3, 0.9), 'Fp2': (0.3, 0.9), 'F3': (-0.3, 0.5),
    'F4': (0.3, 0.5), 'F7': (-0.6, 0.6), 'F8': (0.6, 0.6),
    'P3': (-0.3, -0.5), 'P4': (0.3, -0.5),
    'T5': (-0.6, -0.6), 
    'T6': (0.6, -0.6), 
    'O1': (-0.3, -0.9), 'O2': (0.3, -0.9)
}


sfreq = 128  
ch_names = ["Fp1", "Fp2",
    "F3", "F4","C3","C4", "P3", "P4","O1", "O2", "F7", "F8","T3",  "T4","T5", "T6", "Fz", "Cz", "Pz"  ] 
duration = 5.0  
overlap = 0.5  


ch_names_ordered = ["Fp1", "Fp2",
    "F3", "F4","C3","C4", "P3", "P4","O1", "O2", "F7", "F8","T3",  "T4","T5", "T6", "Fz", "Cz", "Pz"  ]


ch_names = ch_names_ordered

grid_height, grid_width = 8, 8 




def load_eeg_data_from_mat(file_path):
    mat = scipy.io.loadmat(file_path)
    for key in mat:
        if isinstance(mat[key], np.ndarray) and mat[key].ndim >= 2 and mat[key].shape[0] > 10 and mat[key].shape[1] > 10:
            eeg_data = mat[key]
            if eeg_data.shape[0] < eeg_data.shape[1] and eeg_data.shape[0] == len(ch_names):
                eeg_data = eeg_data.T
            return eeg_data
    raise ValueError(f"Could not find suitable EEG data array in {file_path}")

def create_mne_raw(eeg_data, sfreq, ch_names):
    eeg_data = eeg_data.T
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(eeg_data, info)
    return raw

def make_epochs(raw, duration, overlap):
    epochs = mne.make_fixed_length_epochs(raw, duration=duration, overlap=overlap, preload=True)
    X = epochs.get_data(units='V')
    return X

def interpolate_to_grid(epoch_data_2d, electrode_coords, ch_names_order, grid_height, grid_width):
    points = np.array([electrode_coords[ch_name] for ch_name in ch_names_order])
    grid_x, grid_y = np.mgrid[-1:1:grid_width*1j, -1:1:grid_height*1j]

    interpolated_epoch = np.zeros((grid_height, grid_width, epoch_data_2d.shape[1]))

    for t in range(epoch_data_2d.shape[1]):
        values = epoch_data_2d[:, t]
        grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')
        grid_z[np.isnan(grid_z)] = 0.0
        interpolated_epoch[:, :, t] = grid_z

    return interpolated_epoch




data_folder_adhd_1 = "/input/adhd-1"
data_folder_adhd_2 = "/input/adhd-2"
data_folder_control_1="/input/control-1"
data_folder_control_2="/input/control-2"
mat_files_adhd = glob.glob(os.path.join(data_folder_adhd_1, 'v*')) + glob.glob(os.path.join(data_folder_adhd_2, 'v*'))
mat_files_control = glob.glob(os.path.join(data_folder_control_1, 'v*')) + glob.glob(os.path.join(data_folder_control_2, 'v*'))

X_subjects_list = []
y_subjects_list = []




for file_path in mat_files_adhd:
    eeg_data_raw = load_eeg_data_from_mat(file_path)
    raw = create_mne_raw(eeg_data_raw, sfreq, ch_names)
    epochs_data = make_epochs(raw, duration=duration, overlap=overlap)

    processed_epochs_for_subject = []

    for epoch in epochs_data:
   
        interpolated_raw_epoch = interpolate_to_grid(epoch, electrode_coords_10_20, ch_names, grid_height, grid_width)

    

        interpolated_eeg_grid = np.expand_dims(interpolated_raw_epoch, axis=-1)

        processed_epochs_for_subject.append(interpolated_eeg_grid)
   

    if processed_epochs_for_subject:
        X_subjects_list.append(np.array(processed_epochs_for_subject))
        y_subjects_list.append(np.array([1] * len(processed_epochs_for_subject)))

for file_path in mat_files_control:
    eeg_data_raw = load_eeg_data_from_mat(file_path)
    raw = create_mne_raw(eeg_data_raw, sfreq, ch_names)
    epochs_data = make_epochs(raw, duration=duration, overlap=overlap)

    processed_epochs_for_subject = []

    for epoch in epochs_data:
        interpolated_raw_epoch = interpolate_to_grid(epoch, electrode_coords_10_20, ch_names, grid_height, grid_width)
        

    

        interpolated_eeg_grid = np.expand_dims(interpolated_raw_epoch, axis=-1)
        processed_epochs_for_subject.append(interpolated_eeg_grid)
   

 

    if processed_epochs_for_subject:
        X_subjects_list.append(np.array(processed_epochs_for_subject))
        y_subjects_list.append(np.array([0] * len(processed_epochs_for_subject)))

all_subjects_X = []
all_subjects_y = []
subject_ids = []

for i, subject_epochs in enumerate(X_subjects_list):
    all_subjects_X.extend(subject_epochs)
    all_subjects_y.extend(y_subjects_list[i])
    subject_ids.extend([i] * len(subject_epochs))


subject_ids = np.array(subject_ids)
output_dir = "working/"
os.makedirs(output_dir, exist_ok=True)
np.savez(os.path.join(output_dir, 'preprocessed_eeg_data_3D.npz'),
         X=all_subjects_X, y=all_subjects_y, subject_ids=subject_ids)


