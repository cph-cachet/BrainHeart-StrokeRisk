# BrainHeart-StrokeRisk

This repository provides the code and resources for estimating **biological age** through **brain-heart interactions (BHI)** and evaluating **stroke risk** based on EEG and ECG data from the Sleep Heart Health Study (SHHS).

## Overview
The goal of this project is to utilize **polysomnography (PSG)** signals, specifically **EEG** and **ECG**, to estimate a patient’s biological age using a **local lead-attention model**. By comparing the predicted biological age with the chronological age, we assess stroke risk. Our results demonstrate that **joint brain-heart interactions** significantly outperform single-modality approaches.


### Prerequisites
Ensure you have the following dependencies installed:
- **Python** >= 3.9
- **PyTorch**

  
### Dataset
To train and evaluate the model, you need the **SHHS dataset**.

1. **Download the dataset**:  
   Access the SHHS dataset from the National Sleep Research Resource:  
   [Sleep Heart Health Study (SHHS)](https://sleepdata.org/datasets/shhs)

2. **Preprocess the data**:  
   Run the preprocessing script to filter and divide the PSG signals into epochs.
   ```bash
   python preprocess/filter_divide_epochs_shhs_v4.py
   ```

## Usage

### 1. Preprocess Data
Run the preprocessing script to prepare the SHHS dataset:
```bash
python preprocess/filter_divide_epochs_shhs_v4.py
```
This will generate preprocessed EEG and ECG signals divided into epochs.

### Training
To train the local lead-attention model on the preprocessed data:
```
python train.py --measure both
```
Also train for measures *ecg* and *eeg*
Training requires you to download the dataset first and preprocess it with *preprocess/filter_divide_epochs_shhs_v4.py*

### Testing
The predictions for each measure are stored in the folder *outputs*.
Use the following command to generate the pdf to analyse the results - This works without downloading additional data as the predictions are stored in this repo.
```
python generate_report.py
```
Also use the *results.ipynb* to see additional analysis.

Link to download raw data: [https://sleepdata.org/datasets/shhs](https://sleepdata.org/datasets/shhs)

# Contact
For any questions or collaboration inquiries, please reach out to:
- **Gouthamaan Manimaran**: [gouma@dtu.dk](mailto:gouma@dtu.dk)
- **Matteo Saibene**: [saima@dtu.dk](mailto:saima@dtu.dk)


# References
```
@inproceedings{manimaran2023reading,
  title={Reading Between the Leads: Local Lead-Attention Based Classification of Electrocardiogram Signals},
  author={Manimaran, Gouthamaan and Puthusserypady, Sadasivan and Dominguez, Helena and Bardram, Jakob E},
  booktitle={2023 Computing in Cardiology (CinC)},
  volume={50},
  pages={1--4},
  year={2023},
  organization={IEEE}
}
```
```
@article{quan1997sleep,
  title={The sleep heart health study: design, rationale, and methods},
  author={Quan, Stuart F and Howard, Barbara V and Iber, Conrad and Kiley, James P and Nieto, F Javier and O'Connor, George T and Rapoport, David M and Redline, Susan and Robbins, John and Samet, Jonathan M and others},
  journal={Sleep},
  volume={20},
  number={12},
  pages={1077--1085},
  year={1997},
  publisher={Oxford University Press}
}
```
