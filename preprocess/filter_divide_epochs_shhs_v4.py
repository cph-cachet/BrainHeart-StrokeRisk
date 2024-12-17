#%%
import numpy as np
import matplotlib.pyplot as plt
import os
import mne
from pathlib import Path
import pandas as pd
# from Displayer.Printer import Printer
# from Helper.Loader import Loader
import neurokit2 as nk
import xmltodict
import warnings
import datetime

# %% warnings.filterwarnings("ignore")  # Ignore warnings

debug = False
remove_awake = False
plot = False

# base path of dataset
base_path = Path('/work3/saima/shhs')
# base_path = Path('C:\\Users\\matte\OneDrive - Danmarks Tekniske Universitet\DATABASES\shhs')
shh1_data_path = os.path.join(base_path, 'polysomnography', 'edfs', 'shhs1')
shh1_events_path = os.path.join(base_path, 'polysomnography', 'annotations-events-nsrr', 'shhs1')


# get all files in the directory-------------
def get_all_files(directory):
    file_list = []

    # Walk through the directory and subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            # trim the file name adn get subject only
            file = file.split('-', 1)[-1] if '-' in file else file
            file = file.split('.', 1)[0] if '.' in file else file
            # print(file)
            # file = str(file)
            # file = int(file)
            # Append the full file path to the list
            file_list.append(os.path.join(file))

    return file_list


all_files = get_all_files(shh1_data_path)
all_files = np.array(all_files)
# convert elements to numbers
all_files = np.array([int(i) for i in all_files])
# Generate list of files
cvd_summary = pd.read_csv(os.path.join(base_path, 'datasets', 'shhs-cvd-summary-dataset-0.21.0.csv'))
#%%
# can you find all the files in the folder?
file_list = os.listdir(shh1_data_path)
annotation_list = os.listdir(shh1_events_path)

# check if subject is in the list. If not, remove it from healthy_at_baseline
# is healthy_at_baseline in the list of files?
final_list_subjects = []
for sub in all_files:
    if sub in all_files:
        # print(f'{sub} is in the list of all files')
        final_list_subjects.append(sub)



# %%
# Remember to change path
newpath = Path('/work3/saima/shhs_processed/shhs1_processed_all_directories_V4') # path to save the data
#%%

for sub in range(len(final_list_subjects)):  # --- run in loop in final
    print('--------------TIME-----------------', datetime.datetime.now())
    print('Processing subject number:', sub)
    # subject = healthy_at_baseline['nsrrid'].values[sub]
    subject = final_list_subjects[sub]
    print('Loading subject edf:', subject)
 
    # subejct_path = os.path.join(Path('/work3/saima/shhs_processed/shhs1_processed_all_directories_V4'), f'shhs1_{subject}')
    subejct_path = os.path.exists(os.path.join(newpath, f'shhs1_{subject}'))
        # if already processed: skip
    if subejct_path == True:
        print('----------sanity-----------', subejct_path)
        print(f'subject {sub} already processed... skipping to next') #? maybe not the best way....

    # if not os.path.exists(subejct_path):
    if subejct_path == False:
        
        print('directory does not exist, therefore processing subject:', subject)
        # print(f'saving files to {newpath}')
        
        # Load raw data
        raw = mne.io.read_raw_edf(os.path.join(shh1_data_path, f'shhs1-{subject}.edf'), preload=True, verbose=debug)
        print('Loading subject annotation in xml for subject:', subject)

        # Load events in xml file
        event_content = open(os.path.join(shh1_events_path, f'shhs1-{subject}-nsrr.xml')).read()
        # change xml format to ordered dictionart
        event_dict = xmltodict.parse(event_content)['PSGAnnotation']['ScoredEvents']['ScoredEvent']
        # Ordered dictionary to dataframe
        event_df = pd.DataFrame(event_dict)
        event_df['EventConcept'] = event_df['EventConcept'].str.split('|').str[0]
        event_df['EventType'] = event_df['EventType'].str.split('|').str[0]
        event_df['Start'] = pd.to_numeric(event_df['Start'], downcast='integer', errors='coerce')
        event_df['Duration'] = pd.to_numeric(event_df['Duration'], downcast='integer', errors='coerce')
        sleep_stages_df = event_df.loc[event_df['EventType'] == 'Stages'].copy()

        if remove_awake:
            sleep_stages_df = sleep_stages_df[sleep_stages_df['EventConcept'] != 'Wake']

        # preprocess raw data object
        '''
        Linked EEG: Label EEG (sec) electrodes C3 A2 it is named 'EEG(sec)' 'EEG 2' or 'EEG2' or 'EEG sec' or 'EEG(SEC)'
        sometimes EEG (sec) and EEG are flipped. That's why I have the flipped variable when saving the data
        Linked EEG: Label EEG       electrodes C4 A1
        ECG and EEG are sampled at 125 Hz in SHHS1
        '''
        # 1: drop channels
        raw.drop_channels([item for item in raw.ch_names if item not in ['EEG', 'EEG(sec)' , 'EEG 2', 'EEG2' , 'EEG sec', 'EEG(SEC)', 'ECG']], on_missing='warn')
        # 2: Filter with notch filter all signals
        freqs = 50
        raw.notch_filter(freqs=freqs, verbose=debug)
        # pick EEG channels. Pick all channels named 'EEG 2' or 'EEG2' or 'EEG sec' or 'EEG(SEC)' 
        eeg_picks = [item for item in raw.ch_names if item in ['EEG', 'EEG(sec)' , 'EEG 2', 'EEG2' , 'EEG sec', 'EEG(SEC)']]
        # and pick ECG channels
        ecg_pick = [item for item in raw.ch_names if item in ['ECG']]
        # bandpass EEG
        raw.filter(l_freq=1, h_freq=45, picks=eeg_picks, filter_length='auto', phase="zero-double", verbose=debug)
        # bandpass ECG
        raw.filter(l_freq=4, h_freq=45, picks='ECG', filter_length='auto', phase="zero-double", verbose=debug)


        #! Check if EEG(sec) and EEG are flipped. If yes, flip EEG channels and put alwatys the same order
        # Check where the EEG channels are
        ch_list_temp = raw.ch_names
        flipped = False
        if 'EEG' == ch_list_temp[0]:
            print('EEG is in the first position------------------------------------------Flip the order')
            flipped = True #* this is used when saving the data


        if plot:
            raw.plot()
            spectrum = raw.compute_psd()
            spectrum.plot(dB=True, show=True)

        '''Bind annotation to raw object'''
        epoch_duration = 60
        test = raw.copy()
        annotations = mne.Annotations(
            onset=sleep_stages_df['Start'].values,
            duration=sleep_stages_df['Duration'].values,
            description=sleep_stages_df['EventConcept'].values)
        test.set_annotations(annotations)  # set annotations to raw object
        epochs = mne.Epochs(test, tmin=-epoch_duration / 2, tmax=epoch_duration / 2, baseline=None, preload=True,
                            verbose=debug)
        if debug:
            print(epochs.event_id)
        if plot:
            epochs.plot(n_epochs=10, events=True)

        # save data
        # newpath = Path('Downloads/shhs_process')
        # newpath = Path('/work3/saima/shhs_processed/shhs1_processed_all_directories')
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        subejct_path = newpath
        subejct_path = os.path.join(newpath, f'shhs1_{subject}')
        print(f'saving files to {subejct_path}')
        if not os.path.exists(subejct_path):
            os.makedirs(subejct_path)

        # saving stage 1 sleep
        if 'Stage 1 sleep' in sleep_stages_df['EventConcept'].values:
            stage1_epochs = epochs["Stage 1 sleep"].get_data(copy=True)
            if flipped == True:
                print('----------------Swapping dimensions--------------------')
                stage1_epochs[:, [0, 2], :] = stage1_epochs[:, [2, 0], :]
            np.save(os.path.join(subejct_path, 'stage1.npy'), stage1_epochs)
        else:
            print("Stage 1 sleep not found! What a rough night!")

        # saving stage 2 sleep
        if 'Stage 2 sleep' in sleep_stages_df['EventConcept'].values:
            stage2_epochs = epochs["Stage 2 sleep"].get_data(copy=True)
            if flipped == True:
                print('----------------Swapping dimensions--------------------')
                stage2_epochs[:, [0, 2], :] = stage2_epochs[:, [2, 0], :]         
            np.save(os.path.join(subejct_path, 'stage2.npy'), stage2_epochs)
        else:
            print("Stage 2 sleep not found! What a rough night!")

        # saving stage 3 sleep
        if 'Stage 3 sleep' in sleep_stages_df['EventConcept'].values:
            stage3_epochs = epochs["Stage 3 sleep"].get_data(copy=True)
            if flipped == True:
                print('----------------Swapping dimensions--------------------')
                stage3_epochs[:, [0, 2], :] = stage3_epochs[:, [2, 0], :]
            np.save(os.path.join(subejct_path, 'stage3.npy'), stage3_epochs)
        else:
            print("Stage 3 sleep not found! What a rough night!")

        # saving stage 4 sleep
        if 'Stage 4 sleep' in sleep_stages_df['EventConcept'].values:
            stage4_epochs = epochs["Stage 4 sleep"].get_data(copy=True)
            if flipped == True:
                print('----------------Swapping dimensions--------------------')
                stage4_epochs[:, [0, 2], :] = stage4_epochs[:, [2, 0], :]            
            np.save(os.path.join(subejct_path, 'stage4.npy'), stage4_epochs)
        else:
            print("Stage 4 sleep not found! What a rough night!")

        # saving wake
        if 'Wake' in sleep_stages_df['EventConcept'].values:
            wake_epochs = epochs["Wake"].get_data(copy=True)
            if flipped == True:
                print('----------------Swapping dimensions--------------------')
                wake_epochs[:, [0, 2], :] = wake_epochs[:, [2, 0], :]   
            np.save(os.path.join(subejct_path, 'wake.npy'), wake_epochs)
        else:
            print(f"I guess {subject} is dead")

        # saving REM sleep
        if 'REM sleep' in sleep_stages_df['EventConcept'].values:
            rem_epochs = epochs["REM sleep"].get_data(copy=True)
            if flipped == True:
                print('----------------Swapping dimensions--------------------')
                rem_epochs[:, [0, 2], :] = rem_epochs[:, [2, 0], :]                
            np.save(os.path.join(subejct_path, 'rem.npy'), rem_epochs)
        else:
            print("REM sleep not found! What a rough night!")

    continue

print('done----------------------------')
