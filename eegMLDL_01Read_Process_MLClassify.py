#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 12:06:22 2023
Learning Exercise: EEG ML and DL Methods 
Tutorial: https://www.youtube.com/playlist?list=PLtGXgNsNHqPTgP9wyR8pmy2EuM2ZGHU5Z

01. EEG read signal, process and Machine Learning classification using PYTHON
Dataset: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0188629

Content:
    

@author: abinjacob
"""

#%% Libraries 
import mne 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os 

# to read all files from a folder
from glob import glob

#%% import all the files in the folder 
folder_path = '/Users/abinjacob/Documents/CODING FILES/EEG ML DL/dataverse_files'
glob(folder_path + '/*')

#%%
