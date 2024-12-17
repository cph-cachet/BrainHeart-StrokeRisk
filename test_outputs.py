import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import json
import os
from scipy import stats
from scipy.signal import butter, lfilter, resample
from scipy.interpolate import interp1d
import random

from locallead import LocalLeadModel

measure = 'eeg'   # Do for 'ecg', 'eeg', 'both'

path = ''
model_path = ''
batch_size = 8

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = LocalLeadModel(channels=2).to(device)
model.load_state_dict(torch.load(model_path)['model_state_dict'])
model.eval()


def butter_bandpass(lowcut, highcut, fs, order=2):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')


def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def my_norm(a, data_min, data_max):
    ratio = 2/(data_max - data_min)
    shift = (data_max + data_min)/2
    return (a - shift)*ratio


target_fs = 256


class CustomDataset(torch.utils.data.Dataset):
    # measure - 'ecg', 'eeg', 'both'
    def __init__(self, data_path, subjects, measure='both', resample_to=3750, train=True):
        self.data_path = data_path
        self.subjects = subjects
        self.measure = measure
        self.resample_to = resample_to
        self.train = train

        # Calculate age normalization parameters from training data
        if train:
            ages = [subject[4] for subject in subjects]
            self.age_min = min(ages)
            self.age_max = max(ages)

            # Get unique patients
            unique_patients = list(set([subject[0] for subject in subjects]))
            # Randomly sample 100 patients
            random.seed(42)  # for reproducibility
            sampled_patients = random.sample(unique_patients, 100)

            # Calculate signal normalization parameters from 100 random patients
            all_data = []
            for patient in sampled_patients:
                patient_data = []
                patient_samples = [s for s in subjects if s[0]
                                   == patient][:10]  # limit samples per patient
                for sample in patient_samples:
                    data = np.load(os.path.join(
                        data_path, 'shhs1_'+str(sample[0]), sample[1]))[sample[2]]
                    # Get middle 2000 samples
                    start_idx = (data.shape[1] - self.resample_to) // 2
                    data = data[:, start_idx:start_idx+self.resample_to]
                    patient_data.append(data)
                if patient_data:
                    all_data.extend(patient_data)

            all_data = np.array(all_data)
            self.data_min = np.min(all_data, axis=(0, 2))  # Shape: (3,)
            self.data_max = np.max(all_data, axis=(0, 2))  # Shape: (3,)

    def set_normalization_params(self, age_min, age_max, data_min, data_max):
        # For test set, use training set parameters
        self.age_min = age_min
        self.age_max = age_max
        self.data_min = data_min
        self.data_max = data_max

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        file_name = str(self.subjects[idx][0])
        label = self.subjects[idx][3]
        data = np.load(os.path.join(
            self.data_path, 'shhs1_'+str(file_name), self.subjects[idx][1]))[self.subjects[idx][2]]
        # Get middle 2000 samples
        start_idx = (data.shape[1] - self.resample_to) // 2
        data = data[:, start_idx:start_idx+self.resample_to]

       # data = resample(data, self.resample_to, axis=1)

        for i in range(data.shape[0]):  # Iterate over the 3 features
            data[i] = (data[i] - self.data_min[i]) / \
                (self.data_max[i] - self.data_min[i])

        # Normalize age to [0, 1] using training set parameters
        age = self.subjects[idx][4]
        age_normalized = (age - self.age_min) / (self.age_max - self.age_min)

        if self.measure == 'eeg':
            data = data[[0, 2]]
        elif self.measure == 'ecg':
            data = data[1:2]
        elif self.measure == 'both':
            pass
        else:
            raise ValueError('Invalid measure')

        return np.expand_dims(data, 0), age, label, self.subjects[idx]


print('Loading training data')

with open(os.path.join('preprocess', 'train_splits.json'), 'r') as f:
    splits = json.load(f)
    train_array = splits['train']

subjects = np.unique(np.array(train_array)[:, 0])

total_files = len(os.listdir(path))
train_dataset = CustomDataset(path, subjects=train_array, measure=measure)

with open(os.path.join('preprocess', 'test_splits.json'), 'r') as f:
    splits = json.load(f)
    remaining_array = splits['remaining']

subjects = np.unique(np.array(remaining_array)[:, 0])

total_files = len(os.listdir(path))
test_dataset = CustomDataset(path, subjects=remaining_array, measure=measure)

# Pass training set normalization parameters to test set
test_dataset.set_normalization_params(
    test_dataset.age_min,
    test_dataset.age_max,
    test_dataset.data_min,
    test_dataset.data_max
)

# Create train and test dataloaders
test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=True
)


all_preds, all_gts, all_subjects, all_labels, all_sleep_stages = [], [], [], [], []

print('Generating predictions for remaining subjects')

for data, age, label, subject in tqdm(test_dataloader):
    data = data.to(device)
    with torch.no_grad():
        output = model(data.float())
    agePred = output.cpu().numpy() * (train_dataset.age_max -
                                      train_dataset.age_min) + train_dataset.age_min
    all_preds.extend(agePred.reshape(-1).tolist())
    all_gts.extend(age.reshape(-1).tolist())
    all_subjects.extend(subject[0].tolist())
    all_labels.extend(label.tolist())
    all_sleep_stages.extend(np.array(subject[1]).tolist())

np.savez(os.path.join('outputs', f'test_outputs_remaining_{measure}.npz'),
         predictions=np.array(all_preds),
         ground_truths=np.array(all_gts),
         subjects=np.array(all_subjects),
         labels=np.array(all_labels),
         sleep_stages=np.array(all_sleep_stages))
