#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 19:44:55 2023

Learning Exercise: EEG ML and DL Methods 
Tutorial: https://www.youtube.com/playlist?list=PLtGXgNsNHqPTgP9wyR8pmy2EuM2ZGHU5Z

4.0 Read Motor Imagery EEG Signal
Dataset: BCI Competition IV 2a - A01T.gdf
Filepath: /Users/abinjacob/Documents/02. NeuroCFN/Coding Practice/Coding/MI Classification/BCICIV_2a_gdf

@author: abinjacob
"""

#%% Library
import mne 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 

#%% load file

filepath = "/Users/abinjacob/Documents/02. NeuroCFN/Coding Practice/Coding/MI Classification/BCICIV_2a_gdf/A01T.gdf"

# reading file in mne
raw = mne.io.read_raw_gdf(filepath, eog= ['EOG-left', 'EOG-central', 'EOG-right'])
# removing EOG channels 
raw.drop_channels(['EOG-left', 'EOG-central', 'EOG-right'])

samplingFrequency = raw.info['sfreq']

#%% setup the events
# annotations marked in the EEG file for different events
raw.annotations

# creating events
events = mne.events_from_annotations(raw)

# event dictionary
events_dict = {
    'reject' : 1,
    'eye move': 2,
    'eye open': 3,
    'eye close': 4,
    'new run': 5,
    'new trial': 6,
    'class 1': 7,
    'class 2': 8,
    'class 3': 9,
    'class 4': 10,}

#%% plotting the events
fig = mne.viz.plot_events(events[0], event_id= events_dict, sfreq= samplingFrequency, first_samp= raw.first_samp)

#%% epoching 
# considering only class 1,2,3,4 events (event id 7,8,9,10)

epochs = mne.Epochs(raw, events[0], event_id= [7,8,9,10], tmin= -0.1, tmax= 0.7, preload= True)

# checking the shape of the epochs
epochs.get_data().shape

# creating labels for classification (stored in the last column of epochs)
labels = epochs.events[:, -1]

#%% calculating evoked potentials for all of the classes

evokClass_1 = epochs['7'].average()
evokClass_2 = epochs['8'].average()
evokClass_3 = epochs['9'].average()
evokClass_4 = epochs['10'].average()

# creating dict for the classes and evoked events
evokedDict = {
    'Class 1 / left': evokClass_1,
    'Class 2 / right': evokClass_2,
    'Class 3 / foot': evokClass_3,
    'Class 4 / tongue': evokClass_4,}

# plotting evoked potentials 
mne.viz.plot_compare_evokeds(evokedDict)




