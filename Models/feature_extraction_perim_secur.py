# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 18:14:53 2025

"""
import sys
import logging

# Importing necessary modules for data loading and transformation
from data_loader import DASDataLoader, fft

logging.basicConfig(level=logging.INFO)

import pywt
from sklearn.preprocessing import StandardScaler

import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
import pandas as pd
from scipy.signal import hilbert
import pywt
import librosa

def rdft_preprocess(signal, redundancy_factor=3, trim_samples=2580):
    pad_length = len(signal) * redundancy_factor
    fft_result = np.fft.fft(signal, n=pad_length)
    magnitude = np.abs(fft_result[:pad_length // 2])
    log_mag = np.log10(magnitude + 1e-8).astype(np.float32)
    log_mag -= np.mean(log_mag)
    return log_mag[:trim_samples].astype(np.float32)

def dwt_features(signal, wavelet='db4', level=3):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    flattened = np.hstack(coeffs)
    return flattened.astype(np.float32)

def mfcc_features(signal, sr=1000):
    mfccs = librosa.feature.mfcc(
        y=signal.astype(float),
        sr=sr,
        n_mfcc=20,
        dct_type=2,
        n_fft=1024,
        hop_length=128,
        win_length=512,
        window='hann',
        power=2,
        n_mels=128,
        fmin=0,
        fmax=500
    )
    return mfccs.flatten().astype(np.float32)

def feature_extraction(data):  
    def fft_fft(data):
        fft_trans = np.abs(np.fft.fft(data))
        freq_spectrum = fft_trans[1:int(np.floor(len(data) * 1.0 / 2)) + 1]
        _freq_sum_ = np.sum(freq_spectrum)
        return freq_spectrum, _freq_sum_

    
    def rdft_transform(signal, redundancy=3, trim=2580):
        padded = np.zeros(len(signal) * redundancy)
        padded[:len(signal)] = signal
        fft_vals = np.fft.fft(padded)
        mag = np.log10(np.abs(fft_vals) + 1e-8).astype(np.float32)
        mag -= mag.mean()
        return mag[:trim].astype(np.float32)
    
    def dwt_transform(signal, wavelet='db4', level=3):
        coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level)
        return np.concatenate(coeffs).astype(np.float32)
    
    def mfcc_transform(signal, sr=10000):
        mfccs = librosa.feature.mfcc(
            y=signal.astype(float), sr=sr,
            n_mfcc=20, dct_type=2, n_fft=1024,
            hop_length=128, win_length=512,
            window='hann', power=2,
            n_mels=128, fmin=0, fmax=500
        )
        return np.mean(mfccs, axis=1).astype(np.float32)  # mean over time
    
    def extract_combined_features(signal, scalers=None):
        """
        Applique RDFT + DWT + MFCC à un signal 1D.
        scalers: dictionnaire {'rdft':..., 'dwt':..., 'mfcc':...}
        """
        feat_rdft = rdft_transform(signal)
        feat_dwt = dwt_transform(signal)
        feat_mfcc = mfcc_transform(signal)
    
        if scalers is not None:
            feat_rdft = scalers['rdft'].transform(feat_rdft.reshape(1, -1)).flatten()
            feat_dwt = scalers['dwt'].transform(feat_dwt.reshape(1, -1)).flatten()
            feat_mfcc = scalers['mfcc'].transform(feat_mfcc.reshape(1, -1)).flatten()
    
        return np.concatenate([feat_rdft, feat_dwt, feat_mfcc]).astype(np.float32)
    

    

     
    rdft_feat = rdft_preprocess(data)
    dwt_feat = dwt_features(data)
    mfcc_feat = mfcc_features(data)

    combined = np.concatenate([rdft_feat, dwt_feat, mfcc_feat]) #
    return combined

def extract_features_batch(windows):
    feature_list = []
    for i, window in enumerate(windows):
        features = feature_extraction(window)
        feature_list.append(features)
        if i % 100 == 0:
            print(f"Processed {i}/{len(windows)} windows")
    return np.vstack(feature_list)


