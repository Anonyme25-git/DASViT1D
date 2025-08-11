# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 18:14:53 2025

@author: michel
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

    # freq_spectrum, _freq_sum_ = fft_fft(data)
    # analytic_signal = hilbert(data)
    # envelope = np.abs(analytic_signal)
    # # print("max")
    # dif_max = max(data)
    # # print("max feature : ",dif_max)
    # dif_min = min(data)
    # dif_pk = int(dif_max) - int(dif_min)
    # dif_mean = data.mean()
    # dif_median = pd.Series(data).median()
    # dif_skew = pd.Series(data).skew()
    # dif_skew_env = pd.Series(envelope).skew()
    # dif_kurt_env = pd.Series(envelope).kurt()
    # dif_meanFFT = freq_spectrum.mean()
    # dif_maxFFT = freq_spectrum.max()
    # dif_varFFT = freq_spectrum.var()
    # dif_minFFT = freq_spectrum.min()
    # dif_var = data.var()
    # dif_std = data.std()
    # dif_energy = np.sum(freq_spectrum ** 2) / len(freq_spectrum)
    # dif_rms = np.sqrt(pow(dif_mean, 2) + pow(dif_std, 2))
    # dif_arv = abs(data).mean()
    # dif_boxing = dif_rms / (abs(data).mean())
    # dif_maichong = (max(data)) / (abs(data).mean())
    # dif_fengzhi = (max(data)) / dif_rms
    # sum_sqrt = np.sum(np.sqrt(abs(data)))
    # dif_yudu = max(data) / pow(sum_sqrt / len(data), 2)
    # dif_kurt = pd.Series(data).kurt()
    # dif_qiaodu = (np.sum([x ** 4 for x in data]) / len(data)) / pow(dif_rms, 4)
    # pr_freq = freq_spectrum * 1.0 / _freq_sum_
    # dif_entropy = -1 * np.sum([np.log2(p + 1e-5) * p for p in pr_freq])

    # stats_features = np.array([
    #     dif_max, dif_min, dif_pk, dif_mean, dif_energy, dif_var, dif_std, dif_rms,
    #     dif_arv, dif_boxing, dif_maichong, dif_fengzhi, dif_yudu, dif_kurt,
    #     dif_qiaodu, dif_entropy, dif_median, dif_skew, dif_skew_env, dif_kurt_env,
    #     dif_varFFT, dif_meanFFT, dif_minFFT, dif_maxFFT
    # ])
    
    
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
    

    
    # rdft_feat = rdft_transform(data)

    # dwt_feat = dwt_transform(data)
    
    # mfcc_feat = mfcc_transform(data)
     
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



  # ou utiliser `x` après parse_dataset()

# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



# # Paramètres
# window_size = 8192
# keep_bins = 2048
# num_classes = 9
# batch_size = 32
# num_epochs = 20
# learning_rate = 1e-3

# decim_dict = {
#     'car': 1,
#     'longboard': 2,
#     'manipulation': 3,
#     'openclose': 1,
#     'regular': 50,
#     'running': 3,
#     'walk': 2,
#     'fence': 1,  # si tu veux fortement réduire
#     'construction': 3,  # idem
# }

# # 📥 Load dataset using DASDataLoader
# data_dir = "../data"

# loader = DASDataLoader(
#     # data_dir=data_dir,
#     # sample_len=keep_bins,
#     # transform=fft,
#     # fsize=window_size,
#     # shift=2048,
#     # drop_noise=True
    
#     '../data',  # Path to the dataset directory
#     2048,  # Sample length
#     transform=fft,  # Applying FFT as a preprocessing step
#     fsize=8192,  # Window size for sliding window segmentation
#     # Step size for the sliding window (overlap of 75% with fsize=8192)
#     shift=2048,
#     # Dictionary specifying the decimation factor for each label
#     decimate=decim_dict,
# )

# samples, labels = loader.parse_dataset()
# print(f"Loaded dataset: {samples.shape}, Labels: {labels.shape}")


# # === Assumons que tu as déjà : ===
# # samples: np.array de shape (66778, 2048)
# # labels: np.array de shape (66778, 9)

# print("Input samples shape:", samples.shape)
# print("Input labels shape:", labels.shape)

# # Convert one-hot labels to integer indices
# if labels.ndim == 2 and labels.shape[1] > 1:
#     label_indices = np.argmax(labels, axis=1)
# else:
#     label_indices = labels

# print("Converted labels shape:", label_indices.shape)

# # === Appliquer l’extraction ===
# features = extract_features_batch(samples)
# print("Extracted features shape:", features.shape)



# import torch
# import torch.nn as nn

# #coding = UTF-8

# import datetime
# from sklearn import svm, preprocessing
# from get_das_data import get_das_data,get_stats_features
# from sklearn.metrics import confusion_matrix
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.font_manager as fm
# import pandas as pd
# import seaborn as sns


# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler



# # data_total = pd.concat([data_train, data_test], axis=0, ignore_index=True)


# X = features #data_total.iloc[:, :-1].values   
# y = label_indices   


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# # print(f"Shape total: {X.shape}, Shape train: {X_train.shape}, Shape test: {X_test.shape}")


# # Normaliser correctement chaque feature
# # minMaxScaler = preprocessing.MinMaxScaler()
# # X_train = minMaxScaler.fit_transform(X_train)
# # X_test = minMaxScaler.transform(X_test)

# scaler = StandardScaler().fit(X_train)

# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)



# class ViT1D(nn.Module):
#     def __init__(self, input_dim=576, patch_size=96, emb_dim=128, num_heads=4, depth=6, num_classes=6):
#         super(ViT1D, self).__init__()

#         assert input_dim % patch_size == 0, "input_dim must be divisible by patch_size"
#         self.num_patches = input_dim // patch_size
#         self.patch_dim = patch_size

#         # Embedding des patches
#         self.linear_proj = nn.Linear(patch_size, emb_dim)

#         # Position embeddings
#         self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, emb_dim))

#         # Transformer encoder
#         encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, batch_first=True)
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

#         # Classifier
#         self.cls_head = nn.Sequential(
#             nn.LayerNorm(emb_dim),
#             nn.Linear(emb_dim, num_classes)
#         )

#     def forward(self, x):
#         x = x.squeeze(1)  # (batch, seq_len)
#         batch_size, seq_len = x.shape
    
#         # Découper en patches
#         x = x.unfold(1, self.patch_dim, self.patch_dim)  # (batch, n_patches, patch_size)
#         x = x.contiguous()
    
#         x = self.linear_proj(x)  # (batch, n_patches, emb_dim)
    
#         # Adapter pos_embedding dynamiquement 
#         if self.pos_embedding.shape[1] != x.size(1):
#             self.pos_embedding = nn.Parameter(
#                 torch.randn(1, x.size(1), x.size(2), device=x.device)
#             )
    
#         x = x + self.pos_embedding[:, :x.size(1), :]
#         x = self.transformer(x)
#         x = x.mean(dim=1)
#         output = self.cls_head(x)
#         return output
    
#     # + RandomForest
    
#     # def forward(self, x):
#     #     x = x.squeeze(1)
#     #     x = x.unfold(1, self.patch_dim, self.patch_dim).contiguous()
#     #     x = self.linear_proj(x)
    
#     #     if self.pos_embedding.shape[1] != x.size(1):
#     #         self.pos_embedding = nn.Parameter(torch.randn(1, x.size(1), x.size(2), device=x.device))
            
#     #     x = x + self.pos_embedding[:, :x.size(1), :]
#     #     x = self.transformer(x)
#     #     features = x.mean(dim=1)  # (batch, emb_dim)
#     #     return features



# from itertools import product

# vit_param_grid = {
#     'patch_size': [48],
#     'emb_dim': [64],
#     'num_heads': [2],
#     'depth': [3],
#     'lr': [1e-3],
#     'batch_size': [32],
# }

# # Testing combination: {'patch_size': 48, 'emb_dim': 64, 'num_heads': 2, 'depth': 3, 'lr': 0.001, 'batch_size': 32}

# vit_param_names = list(vit_param_grid.keys())
# vit_combinations = list(product(*vit_param_grid.values()))




# # import torch
# import torch.nn as nn
# # import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# # from sklearn.metrics import accuracy_score

# # Convertir les données en Tensor
# X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
# X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
# y_train_tensor = torch.tensor(y_train, dtype=torch.long)
# y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# # Reshape car Conv1D attend (batch_size, channels, sequence_length)
# X_train_tensor = X_train_tensor.unsqueeze(1)  # (batch_size, 1, seq_len)
# X_test_tensor = X_test_tensor.unsqueeze(1)



# # Initialiser modèle
# device = torch.device("cuda")

# def train_and_evaluate_vit(params, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, device):
#     patch_size, emb_dim, num_heads, depth, lr, batch_size = params

#     model = ViT1D(
#         input_dim=576,
#         patch_size=patch_size,
#         emb_dim=emb_dim,
#         num_heads=num_heads,
#         depth=depth,
#         num_classes=6
#     ).to(device)

#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     criterion = nn.CrossEntropyLoss()

#     # DataLoader avec batch_size dynamique
#     train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
#     test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size)

#     for epoch in range(100):  # on garde 10 epochs pour la grid search rapide
#         model.train()
#         for batch_x, batch_y in train_loader:
#             batch_x, batch_y = batch_x.to(device), batch_y.to(device)
#             optimizer.zero_grad()
#             outputs = model(batch_x)
#             loss = criterion(outputs, batch_y)
#             loss.backward()
#             optimizer.step()

#     # Évaluation
#     model.eval()
#     predictions, labels = [], []
#     with torch.no_grad():
#         for batch_x, batch_y in test_loader:
#             batch_x = batch_x.to(device)
#             outputs = model(batch_x)
#             preds = torch.argmax(outputs, dim=1).cpu().numpy()
#             predictions.extend(preds)
#             labels.extend(batch_y.numpy())

#     C = confusion_matrix(labels, predictions)
#     acc = np.trace(C) / np.sum(C)
#     return acc, C





# best_score = 0
# best_params = None
# best_conf = None

# for comb in vit_combinations:
#     print(f"\n🔍 Testing combination: {dict(zip(vit_param_names, comb))}")
#     acc, C = train_and_evaluate_vit(comb, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, device)
#     print(f"🎯 Accuracy: {acc:.4f}")
#     if acc > best_score:
#         best_score = acc
#         best_params = comb
#         best_conf = C

# print("\n✅ Best Accuracy:", best_score)
# print("🔧 Best Params:", dict(zip(vit_param_names, best_params)))
# print("📊 Confusion Matrix:\n", best_conf)