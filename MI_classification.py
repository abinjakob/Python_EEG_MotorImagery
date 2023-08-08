#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 12:50:43 2023

The analysis and classification of Motor-Imagery EEG Data
Dataset: BCI Competition IV Dataset 01 (from Berlin, sampling rate: 100 Hz)
https://www.bbci.de/competition/iv/#dataset1

Description of Data set: https://www.bbci.de/competition/iv/desc_1.html
EEG was recorded from 59 electrodes. The subject was given a cue to imagine 
either righrt or left movement or movement of the feet. 

This code is part of learning to handle MI data from the youtube video:
https://youtu.be/EAQcu6DLAS0


Directry for the file: 
/Users/abinjacob/Documents/03. Calypso/BCI/Motor Imagery/Python for MI classification/data_set_IV


Used Methods:
Common Spatial Pattern (CSP)
Linear Discriminant Analysis (LDA)


@author: abinjacob
"""

#%% libraries 
import numpy as np
from matplotlib import mlab
import matplotlib.pyplot as plt
import scipy.signal
from numpy import linalg

# since we are using matlab file: using the scipy input output library
import scipy.io

#%% load the data (file is in matlab format)
# Dataset: BCI Competition IV Dataset 01 (from Berlin, sampling rate: 100 Hz)
m = scipy.io.loadmat('BCICIV_calib_ds1d.mat', struct_as_record = True)

#%% Arranging the data from the Matlab structure for Python Pandas

# Note: Scipy.io.loadmat does not deal well with Matlab structures, resulting in lots of extra dimenstion 
# in the arrays. Hence need to go through the files well to extract what is required

# extracting the sampling rate
sample_rate = m['nfo']['fs'][0][0][0][0]

# extracting the EEG data (Transposing to get the channels in raws and data in columns for each channels)
EEG = m['cnt'].T
nchannels, nsamples = EEG.shape                 # extracting the number of channels and the data points 

# channel names
channel_names = [s[0] for s in m['nfo']['clab'][0][0][0]]

# extracting event markers 
event_onset = m['mrk'][0][0][0]

# extracting the event ceodes
# -1 : Left Hand Imagery | 1 : Right Hand Imagery
event_codes = m['mrk'][0][0][1] 

# adding labels to the codes
labels = np.zeros((1, nsamples), int)           # an empty array with zeros for the entire data length
labels[0,event_onset] = event_codes             # adding the event codes (-1 or 1) to the time of events 

# ectracting the name of the labels 
cl_lab = [s[0] for s in m['nfo']['classes'][0][0][0]]
cl1 = cl_lab[0]
cl2 = cl_lab[1]

nclasses = len(cl_lab)
nevents  = len(event_onset)

#%% print the details of the recording

print(f'Shape of EEG: {EEG.shape}')
print(f'Sampling rate: {sample_rate}')
print(f'Number of channels: {len(channel_names)}')
print(f'Channel names: {channel_names}')
print(f'Number of events: {len(event_onset)}')
print(f'Event codes: {np.unique(event_codes)}')
print(f'Class labels: {cl_lab}')
print(f'Number of classes: {nclasses}')

#%% setup for looping over the classes 

# dictionary to store the trials, each class gets an entry
trials = {}

# time window (in samples) to extract for each trial (0.5s to 2.5s)
win = np.arange(int(0.5*sample_rate), int(2.5*sample_rate))

# length of time window
nsamples = len(win)

#%% extract the data for left and right for all trials 

# loop over the classes 
# zip() function is used to iterate over cl_lab and np.unique(event_codes) simultaneously. 
# In each iteration, the corresponding elements from cl_lab and np.unique(event_codes) are 
# assigned to cl and code, respectively. 
# np.unique(event_codes) just extracts -1 and 1
for cl, code in zip(cl_lab, np.unique(event_codes)):
    
    # extract the onsets for the class
    cl_onset = event_onset[event_codes == code]
    
    # allocate memory for trials
    trials[cl] = np.zeros((nchannels, nsamples, len(cl_onset)))
    
    # loop over the trials 
    for i, onset in enumerate(cl_onset):
        trials[cl][:,:,i] = EEG[:,win+onset]
        
# here in the trials dictionary it is grouped as left and right
# then the data for each is stored in a 3d matrix (channels x time x trials)
    
#%% function to compute Power Spectral Density (PSD) for each trials 

# since the features we are looking for is the decrease of beta/mu activity, which is 
# a frequency feature, we need PSD of the trials

# creating function
def psd(trials):
    trials_PSD = np.zeros((nchannels,101,trials.shape[2])) 

    # loop over trials 
    for trial in range (trials.shape[2]):
        
        # loop over channels
        for ch in range(nchannels):
            
            # calculate PSD
            (PSD, freqs) = mlab.psd(trials[ch,:,trial], NFFT= int(nsamples), Fs= sample_rate)
            trials_PSD[ch,:,trial] = PSD.ravel()        # ravel is used to flatten multidimensional array to 1D array
            
    # function output
    return trials_PSD, freqs

#%% calculating PSD for right and left

# psd for right hand from left channels
psd_r, freqs = psd(trials[cl1])
# psd for left hand from right channels
psd_l, freqs = psd(trials[cl2])

trials_PSD = {cl1: psd_r, cl2: psd_l}

#%% function to plot the PSD

def plot_psd(trials_PSD, freqs, chan_ind, chan_lab= None, maxy= None): 
# def plot_psd(trials_PSD, freqs, chan_ind):   
    '''
    Plots PSD data calculated with psd()
    
    Parameters
    ----------
    
    trials   : 3d-array (channels x samples x trilas)
        PSD data as returned by psd
    freqs    : list of floats
        Frequencies for which the PSD is defined, as returned by psd()
    chan_ind : list of integers
        Indices of channels to plot
    chan_lab : list of strings
        (optional) List of names for each channel
    maxy     : float
        (optional) Ylim
    '''
    
    plt.figure(figsize= (12,5))
    
    # number of channles
    nchans = len(chan_ind)
    
    # maximum of 3 plots per row
    nrows = np.ceil(nchans/3).astype(int)
    ncols = min(3, nchans)
    
    # ennumerate over channels
    for i, ch in enumerate(chan_ind):
        
        # choosing subplot to draw
        plt.subplot(nrows, ncols, i+1)
        
        # plot PSD for each class
        for cl in trials.keys():
            plt.plot(freqs, np.mean(trials_PSD[cl][ch,:,:], axis= 1), label= cl)
        
        # plot decorations 
        plt.xlim(1,30)
        
        if maxy != None:
            plt.ylim(0,maxy)
            
        plt.grid()
        plt.xlabel('frequency [Hz]')
        
        if chan_lab == None:
            plt.title(f'Channel {ch+1}')
        else:
            plt.title(chan_lab[i])

        plt.legend()
    plt.tight_layout()

#%% plotting PSD for C3, C4 and Cz
        
plot_psd(
    trials_PSD,
    freqs,
    [channel_names.index(ch) for ch in ['C3', 'Cz','C4']],
    chan_lab = ['C3', 'Cz', 'C4'],
    maxy = 500
    )

#%% NOTES

# After plotting the PSD we can see the differences in left and right channels during the hand movement. 
# These differences will be used as input features to the classifier. 
# We will apply ML to classify the Left hand and Right hand movements. 
# Before that there are few steps to be performed:
#    1. Need to quantify the amount of mu activity present in each trial
#    2. Make a model that descreibes the expected value of mu activity for each class
#    3. Test the model on an unseen data to see if it can predict these class labels correctly
# 
# A classic BCI design by Blankertz (2007), ie. log of variance of the signal in mu band, as the feature for the  classifier

#%% function for bandpass filter

def bandpass(trials, lo, hi, sample_rate):
    '''
    Designs and applies a bandpass filter to the signal

    Parameters
    ----------
    trials : 3d-array (channels x samples x trilas)
        EEG signal
    lo : float
        Lower frequency bound (Hz)
    hi : float
        upper frequency bound (Hz)
    sample_rate : float
        Sampling rate (Hz)

    Returns
    -------
    trial_filt : 3d-array (channels x samples x trials)
        Bandpassed signal

    '''
    
    # sampling rate/2 is done to satisfy nyquist theory
    a, b = scipy.signal.iirfilter(6, [lo/(sample_rate/2.0), hi/(sample_rate/2.0)])
    
    # applying filter to each trials 
    ntrials = trials.shape[2]
    trials_filt = np.zeros((nchannels, nsamples, ntrials))
    
    # loop over trials 
    for i in range(ntrials):
        trials_filt[:,:,i] = scipy.signal.filtfilt(a, b, trials[:,:,i], axis= 1)
    
    return trials_filt

#%% Applying the bandpass filter to the data 

# filtering just the mu band (8-15 Hz) from left and right sinals and store in a dict
trials_filt = {cl1: bandpass(trials[cl1], 8, 15, sample_rate),
               cl2: bandpass(trials[cl2], 8, 15, sample_rate)}
#%% applying PSD to mu band data

# psd for right hand from left channels
psd_r, freqs = psd(trials_filt[cl1])
# psd for left hand from right channels
psd_l, freqs = psd(trials_filt[cl2])

trials_PSD = {cl1: psd_r, cl2: psd_l}

# plot PSD for C3, Cz and C4
plot_psd(
    trials_PSD,
    freqs,
    [channel_names.index(ch) for ch in ['C3', 'Cz','C4']],
    chan_lab = ['C3', 'Cz', 'C4'],
    maxy = 300
    )

#%% function to calculate the log(variance) for each trials

def logvar(trials):
    '''
    Calculate the log-var of each channel

    Parameters
    ----------
    trials : 3d-array (channels x samples x trilas)
        EEG signal

    Returns
    -------
    logvar : 2d-array (channels x trials)
        For each channel the logvar of the signal

    '''
    return np.log(np.var(trials, axis= 1))

#%% calculating logvar() to bandpassed signal

trials_logvar = {cl1: logvar(trials_filt[cl1]),
                cl2: logvar(trials_filt[cl2])}

#%% function to visualise the logvar in a bar chart 
def plot_logvar(trials):
    
    plt.figure(figsize= (12,5))
    
    x0 = np.arange(nchannels)
    x1 = np.arange(nchannels) + 0.4
    
    y0 = np.mean(trials[cl1], axis= 1)
    y1 = np.mean(trials[cl2], axis= 1)
    
    plt.bar(x0, y0, width= 0.5, color= 'b')
    plt.bar(x1, y1, width= 0.4, color= 'r')
    
    plt.xlim(-0.5, nchannels + 0.5)
    
    plt.gca().yaxis.grid(True)
    plt.title('Log-var of each channel')
    plt.xlabel('channels')
    plt.ylabel('log-var')
    plt.legend(cl_lab)
    
#%% plotting log-var
plot_logvar(trials_logvar)

# Note: In the plot we can see small difference between the channels. However, we need to maximise differences
# We use Common Spatial Pattern (CSP) to maximise the variance between the two classes

#%% Notes

# to calculate the CSP we need Covariance Matrix of each trial
# also need to perform Singular Value Decomposition (SVD) for Whitening

# CSP
# ---
# CSP is a technique commonly used in brain-computer interface (BCI) applications 
# to extract discriminative features from multichannel electroencephalography (EEG) 
# signals. It aims to find a linear transformation of the original EEG channels to 
# create new channels, called spatial filters, that enhance the differences between 
# two different classes of brain activity.

# Singular Value Decomposition (SVD)
# ----------------------------------
# n the context of Common Spatial Patterns (CSP), SVD is used to decompose the
 # covariance matrix of multichannel signals.

# WHITENING 
# ----------
# Whitening is a preprocessing step often applied before applying CSP. 
# It involves transforming the multichannel data to a new representation where 
# the covariance matrix becomes the identity matrix. This is achieved by using 
# the SVD of the covariance matrix. By whitening the data, the correlations 
# between the channels are removed, and the transformed data has equal variances 
# in all directions.
    
#%% Creating the functions for: Covariance, Whitening, CSP, Apply CSP Weights to data

# function to calculate the covariance for each trials and return the average
def cov(trials):    
    ntrials = trials.shape[2]
    covs = [ trials[:,:,i].dot(trials[:,:,i].T) / nsamples for i in range(ntrials) ]
    
    return np.mean(covs, axis= 0)


# function to calculate a whitening matrix for covariance matrix 'sigma'
def whitening(sigma):
    U, l, _ = linalg.svd(sigma)
    
    return U.dot(np.diag(l ** -0.5))


# calculate CSP transformation matrix W
def csp(trials_r, trials_l):
    '''
    calculate CSP transformation matrix W    

    Parameters
    ----------
    trials_r : 3d-array (channels x samples x trilas) containing right hand movement trials
    trials_l : 3d-array (channels x samples x trilas) containing left hand movement trials

    Returns
    -------
    Mixing matrix W

    '''
    
    # calculate covariance
    cov_r = cov(trials_r)
    cov_l = cov(trials_l)
    
    # whitening
    P = whitening(cov_r + cov_l)
    B, _, _ = linalg.svd(P.T.dot(cov_l).dot(P))
    W = P.dot(B)
    
    return W

# W is the spatial filters which maximise the variance for one class and maximise the variance for other class


# apply CSP Weights to each trials (W * EEG signal matrix)
def apply_mix(W, trials):
    ntrials = trials.shape[2]
    trials_csp = np.zeros((nchannels, nsamples, ntrials))
    
    # loop over trials
    for i in range(ntrials):
        trials_csp[:,:,i] = W.T.dot(trials[:,:,i])
        
    return trials_csp
    
#%% applying the functions 

# applying CSP function to left and right on filtered data
W = csp(trials_filt[cl1], trials_filt[cl2])

# applying the weights 
trials_csp = {cl1: apply_mix(W, trials_filt[cl1]),
              cl2: apply_mix(W, trials_filt[cl2])}

#%% plot the results after CSP

trials_logvar = {cl1: logvar(trials_csp[cl1]),
                 cl2: logvar(trials_csp[cl2])}
plot_logvar(trials_logvar)

# now we can see the  maximised difference between both the classes

#%% visualising PSD after CSP

psd_r, freqs = psd(trials_csp[cl1])
psd_l, freqs = psd(trials_csp[cl2])
trials_PSD = {cl1: psd_r, 
              cl2: psd_l}

plot_psd(trials_PSD, freqs, [0,58,-1], chan_lab= ['first component', 'middle component', 'last  component'], maxy= 0.75)

# there is now a significant difference compared to before CSP
# it is now fed into a classifier to obtain good accuracy

#%% function to create a scatter plot to visualise how well data can be descriminated

def plot_scatter(left, right):
    plt.figure()
    plt.scatter(left[0,:], left[-1,:], color= 'b') 
    plt.scatter(right[0,:], right[-1,:], color= 'r')
    plt.xlabel('last component')
    plt.ylabel('first component')
    plt.legend(cl_lab)

#%% plotting scatter plot
plot_scatter(trials_logvar[cl1], trials_logvar[cl2])

# here as the data can be easily be separated using a straight line, a linear classifier will be used to train the model

#%% dividing data for training (for linear classifier)

# percentage of trials to use for the training (50-50 split)
train_percentage = 0.5

# calculate number of trials for each class for training set
ntrain_r = int(trials_filt[cl1].shape[2] * train_percentage)
ntrain_l = int(trials_filt[cl2].shape[2] * train_percentage)

# calculate number of trials for each class for data set
ntest_r = trials_filt[cl1].shape[2] - ntrain_r
ntest_l = trials_filt[cl2].shape[2] - ntrain_l

# splitting the frequency filtered signal into train set
train = {cl1: trials_filt[cl1][:,:,:ntrain_r],
         cl2: trials_filt[cl2][:,:,:ntrain_l]}

# splitting the frequency filtered signal into test set
test = {cl1: trials_filt[cl1][:,:,ntest_r:],
        cl2: trials_filt[cl2][:,:,ntest_l:]}

#%% CSP on train set
# CSP is a supervised ML training algorithm as we are usning labels

# train the CSP on the train set alone
W = csp(train[cl1], train[cl2])

#%% checking the dimensionality

train[cl1].shape
train[cl2].shape

W.shape

#%% applying CSP to train and test sets
train[cl1] = apply_mix(W, train[cl1])
train[cl2] = apply_mix(W, train[cl2])
test[cl1] = apply_mix(W, test[cl1])
test[cl2] = apply_mix(W, test[cl2])

train[cl1].shape            # will have a matrix of (59,200,50) 

#%% selecting only first and last component for classification
# because the first and last component gives maximum classification

comp = np.array([0,-1])
train[cl1] = train[cl1][comp,:,:]
train[cl2] = train[cl2][comp,:,:]
test[cl1] = test[cl1][comp,:,:]
test[cl2] = test[cl2][comp,:,:]

# no we have only 2 components
train[cl1].shape            # will have a matrix of (2,200,50)  

#%% calculate the log-var

train[cl1] = logvar(train[cl1])
train[cl2] = logvar(train[cl2])
test[cl1] = logvar(test[cl1])
test[cl2] = logvar(test[cl2])

train[cl1].shape            # will have a matrix of (2,50) -> (observation x features)
train[cl1].T

# using this now we can train the classifier 

#%% Linear Discriminant Analysis (LDA)

# LDA algorithm will be used as classifier. It fits a gaussian distribution to 
# each class, characterised by mean and covariance, and determines an optimal 
# separation plane to divide the two. 
    
# function for LDA training 
def train_lda(class1, class2):
    '''
    Trains the LDA algorithm

    Parameters
    ----------
    class1 : array (observation x features) for class 1
    class2 : array (observation x features) for class 2

    Returns
    -------
    projection matrix W
    offset b
    '''
    
    nclasses = 2
    
    nclass1 = class1.shape[0]
    nclass2 = class2.shape[0]
    
    # Class prior: in this case we have an equal number of training examples for
    # each class, so both priors are 0.5
    
    prior1 = nclass1 / float(nclass1 + nclass2)
    prior2 = nclass2 / float(nclass1 + nclass1)
    
    # calculate the mean
    mean1 = np.mean(class1, axis= 0)
    mean2 = np.mean(class2, axis= 0)
    
    class1_centered = class1 - mean1
    class2_centered = class2 - mean2
    
    # calculate the covariance between features
    cov1 = class1_centered.T.dot(class1_centered) / (nclass1 - nclasses)
    cov2 = class2_centered.T.dot(class2_centered) / (nclass2 - nclasses)
    
    # projection matrix
    W = (mean2 - mean1).dot(np.linalg.pinv(prior1 * cov1 + prior2 * cov2))
    # offset
    b = (prior1 * mean1 + prior2 * mean2).dot(W)
    
    return (W,b)



# function for applying LDA to a new data
def apply_lda(test, W, b):
    '''
    applies a previously trained LDA to a new data

    Parameters
    ----------
    test : array (features x trials) containing the data
    W    : projection matrix W as calculated by train_lda()
    b    : offset b as calculated by train_lda()

    Returns
    -------
    list containing a classlabel for each trial

    '''
    ntrials = test.shape[1]
    
    # to store the predictions
    prediction = []
    
    # loop over trials
    for i in range(ntrials):
        # the line below is the generlaisation for:
        # result = W[0] * test[0,i] + W[1] * test[1,i] - b
        
        result = W.dot(test[:,i]) - b
        
        if result <= 0:
            prediction.append(1)
        else:
            prediction.append(2)
    
    return np.array(prediction)


#%% applying training of lDA to training set

W, b = train_lda(train[cl1].T, train[cl2].T)

print(f'W: {W}')
print(f'b: {b}')

#%% plotting scatter plot - Training Data
# plotting the decision boundary with the training data used to calculate it 

plot_scatter(train[cl1], train[cl2])
plt.title('Training Data')

# calculate decision boundary
x = np.arange(-5,0,0.1)    
y = (b - W[0]*x) / W[1]

# plot the decision boundary
plt.plot(x,y, linestyle= '--', linewidth= 2, color= 'k')
plt.xlim(-5,1)
plt.ylim(-2.2,1)

#%% plotting scatter plot - Test Data  

plot_scatter(test[cl1], test[cl2])
plt.title('Test Data')

# plot the decision boundary
plt.plot(x,y, linestyle= '--', linewidth= 2, color= 'k')
plt.xlim(-5,1)
plt.ylim(-2.2,1)  


#%% applying LDA in test and calculating the confusion matrix 

conf = np.array([
    [(apply_lda(test[cl1], W, b) == 1).sum(), (apply_lda(test[cl2], W, b) == 1).sum()],
    [(apply_lda(test[cl1], W, b) == 2).sum(), (apply_lda(test[cl2], W, b) == 2).sum()],
    ])

print('Confusion Matrix')
print(conf)
print(f'Accuracy: {(np.sum(np.diag(conf))/float(np.sum(conf))*100)}%')
    