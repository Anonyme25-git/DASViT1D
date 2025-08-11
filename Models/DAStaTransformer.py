# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 17:01:26 2025

@author: michel
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn

#coding = UTF-8

import datetime

# from get_das_data import get_das_data,get_stats_features
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



import pickle

# Charger les splits
with open("splits/split_indices.pkl", "rb") as f:
    split_dict = pickle.load(f)

train_idx = split_dict['train_idx']
val_idx = split_dict['val_idx']
test_idx = split_dict['test_idx']

# Charger ton dataset spécifique (ex: MFCC ou RDFT)
data_total = pd.read_csv("mfcc__dwt_rdft_feat _data.csv",header=None)
X = data_total.iloc[:, :-1].values
y = data_total.iloc[:, -1].values

# Appliquer les splits
X_train = X[train_idx]
y_train = y[train_idx]
X_val   = X[val_idx]
y_val   = y[val_idx]
X_test  = X[test_idx]
y_test  = y[test_idx]

# Ensuite tu appelles ton modèle
# from models.model_mfcc import create_model


scaler = StandardScaler().fit(X_train)

Xtrain = scaler.transform(X_train)
Xval = scaler.transform(X_val)
Xtest = scaler.transform(X_test)



class ViT1D(nn.Module):
    def __init__(self, input_dim=8192, patch_size=512, emb_dim=128, num_heads=16, depth=6, num_classes=9):
        super(ViT1D, self).__init__()

        assert input_dim % patch_size == 0, "input_dim must be divisible by patch_size"
        self.num_patches = input_dim // patch_size
        self.patch_dim = patch_size

        # Embedding des patches
        self.linear_proj = nn.Linear(patch_size, emb_dim)

        # Position embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, emb_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Classifier
        self.cls_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, num_classes)
        )

    def forward(self, x):
        x = x.squeeze(1)  # (batch, seq_len)
        batch_size, seq_len = x.shape
        
        # self.dropout = nn.Dropout(0.5)
        # Découper en patches
        x = x.unfold(1, self.patch_dim, self.patch_dim)  # (batch, n_patches, patch_size)
        x = x.contiguous()
    
        x = self.linear_proj(x)  # (batch, n_patches, emb_dim)
    
        # Adapter pos_embedding dynamiquement 
        if self.pos_embedding.shape[1] != x.size(1):
            self.pos_embedding = nn.Parameter(
                torch.randn(1, x.size(1), x.size(2), device=x.device)
            )
    
        x = x + self.pos_embedding[:, :x.size(1), :]
        x = self.transformer(x)
        # x = self.dropout(x)
        x = x.mean(dim=1)
        output = self.cls_head(x)
        
        return output
    


# import torch
import torch.nn as nn
# import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# from sklearn.metrics import accuracy_score



# Convertir les données en Tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(Xval, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)

y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Reshape car Conv1D attend (batch_size, channels, sequence_length)
X_train_tensor = X_train_tensor.unsqueeze(1)  # (batch_size, 1, seq_len)
X_test_tensor = X_test_tensor.unsqueeze(1)



# Créer DataLoader 
batch_size = 256
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Initialiser modèle 
train_losses = []
val_losses = []

device = torch.device("cuda")
# from itertools import product
import torch
from sklearn.utils.class_weight import compute_class_weight


# Tes étiquettes de train sous forme d'entiers
unique_classes = np.unique(y_train)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=unique_classes,
    y=y_train
)

# Transformer en Tensor
class_weights = [1.35916111, 0.35037818, 1.35349794, 2.6674777,  0.63500338, 1.70271145, 0.69591896, 2.09273841, 4.55382485]
weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)


model = ViT1D().to(device)

# Définir la perte et l'optimiseur
# criterion = nn.CrossEntropyLoss()
criterion = nn.CrossEntropyLoss(weights_tensor)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Créer les DataLoaders
from torch.utils.data import DataLoader, TensorDataset

train_dataset = TensorDataset(torch.tensor(Xtrain, dtype=torch.float32).unsqueeze(1), 
                              torch.tensor(y_train, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(Xtest, dtype=torch.float32).unsqueeze(1), 
                             torch.tensor(y_test, dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ----------------------
# 🚀 Entraînement
# ----------------------
start_train = datetime.datetime.now()
model.train()
num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Eval sur validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

end_train = datetime.datetime.now()

plt.figure(figsize=(8, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.grid(True)
plt.savefig('train_val_loss_curve.png', dpi=300)
plt.show()


# ----------------------
# 🚀 Test
# ----------------------
start_test = datetime.datetime.now()
model.eval()
predictions = []
labels = []
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        outputs = model(batch_x)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        
        predictions.extend(preds)
        labels.extend(batch_y.numpy())
end_test = datetime.datetime.now()

# ----------------------

# from sklearn.metrics import confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Matrice de confusion absolue
C = confusion_matrix(labels, predictions)

# Conversion en pourcentage ligne par ligne (chaque ligne = 100%)
C_percent = C / C.sum(axis=1, keepdims=True) * 100  # format [%]

# Heatmap style publication
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
df = pd.DataFrame(C_percent)

sns.heatmap(df, fmt='.1f', annot=True, robust=True,
            annot_kws={'size': 12},
            xticklabels=['Car', 'Fence', 'Longboard', 'Manip', 'Manhole', 'Regular', 'Construction', 'Running', 'Walking'],
            yticklabels=['Car', 'Fence', 'Longboard', 'Manip', 'Manhole', 'Regular', 'Construction', 'Running', 'Walking'],
            cmap='Reds',
            linewidths=0.5, linecolor='white', cbar=True)

# plt.title("Confusion Matrix (in %) for Transformer Model")
ax.set_xlabel('Predicted label', fontsize=14)
ax.set_ylabel('True label', fontsize=14)

plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(rotation=0, fontsize=12)

# Personnalisation de la colorbar
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=12)
cbar.set_label("Percentage (%)", fontsize=12)

plt.tight_layout()
plt.savefig(r'C:\Users\michel\Downloads\confusion_matrix_percentage_DASViT1D.png', dpi=300)
plt.show()


# Métriques
Acc = np.trace(C) / np.sum(C)
print('Accuracy: %.4f' % Acc)
NAR = (np.sum(C[5]) - C[5][5]) / np.sum(C[:, [i for i in range(C.shape[1]) if i != 5]])

print('NAR: %.4f' % NAR)
FNR = (np.sum(C[:, 5]) - C[5][5]) / np.sum(C[[i for i in range(C.shape[0]) if i != 5]])
print('FNR: %.4f' % FNR)

column_sum = np.sum(C, axis=0)
row_sum = np.sum(C, axis=1)
print('Column sums:', column_sum)
print('Row sums:', row_sum)

prec =[] 
rec = []
f1 = []
for i in range(0, 9):
    Precision = C[i-1, i-1] / column_sum[i-1] if column_sum[i-1] != 0 else 0
    prec.append(Precision) 
    Recall = C[i-1, i-1] / row_sum[i-1] if row_sum[i-1] != 0 else 0
    rec.append(Recall)
    F1 = 2 * Precision * Recall / (Precision + Recall) if (Precision + Recall) != 0 else 0
    f1.append(F1)
    print(f'Precision_{i}: {Precision:.3f}')
    print(f'Recall_{i}: {Recall:.3f}')
    print(f'F1_{i}: {F1:.3f}')
    
print("Training Time :", end_train - start_train)
print("Inference Time :", end_test - start_test)

precision_avg = np.mean(prec)

recall_avg = np.mean(rec)

f1_avg = np.mean(f1)

# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Matrice de confusion brute (copiée depuis l'image)
# conf_mat = np.array([
#     [287, 21, 5, 3, 15, 16, 1, 20, 34],
#     [15, 200, 1, 6, 4, 0, 7, 2, 22],
#     [7, 8, 533, 1, 14, 4, 0, 23, 3],
#     [8, 6, 4, 492, 30, 3, 0, 11, 14],
#     [13, 21, 7, 11, 440, 7, 1, 27, 48],
#     [9, 4, 3, 2, 9, 469, 1, 13, 24],
#     [0, 8, 0, 0, 0, 0, 407, 0, 2],
#     [10, 2, 8, 5, 20, 9, 0, 353, 21],
#     [19, 34, 0, 8, 20, 21, 2, 18, 515],
# ])

# # Convertir en pourcentages ligne par ligne
# conf_mat_percent = conf_mat / conf_mat.sum(axis=1, keepdims=True) * 100

# # Affichage
# plt.figure(figsize=(10, 8))
# sns.heatmap(conf_mat_percent, annot=True, fmt=".1f", cmap="Blues", cbar=True,
#             xticklabels=['Car', 'Fence', 'Longboard', 'Manip', 'Manhole', 'Regular', 'Construction', 'Running', 'Walking'],
#             yticklabels=['Car', 'Fence', 'Longboard', 'Manip', 'Manhole', 'Regular', 'Construction', 'Running', 'Walking'])
# plt.title("Confusion Matrix (in %) for CNN Model")
# plt.xlabel("Predicted label")
# plt.ylabel("True label")
# plt.xticks(rotation=45, ha='right', fontsize=12)
# plt.yticks(rotation=0, fontsize=12)
# plt.tight_layout()
# plt.show()


"""Training time: 0:12:18.448158
Testing time: 0:00:00.153999
Accuracy: 0.9259
NAR: 0.0109
FNR: 0.0046
Column sums: [239 838 269 100 441 157 408 122  58]
Row sums: [215 835 216 109 461 172 420 140  64]
Precision_0: 0.914
Recall_0: 0.828
F1_0: 0.869
Precision_1: 0.841
Recall_1: 0.935
F1_1: 0.885
Precision_2: 0.967
Recall_2: 0.970
F1_2: 0.968
Precision_3: 0.758
Recall_3: 0.944
F1_3: 0.841
Precision_4: 0.980
Recall_4: 0.899
F1_4: 0.938
Precision_5: 0.957
Recall_5: 0.915
F1_5: 0.936
Precision_6: 0.930
Recall_6: 0.849
F1_6: 0.888
Precision_7: 0.961
Recall_7: 0.933
F1_7: 0.947
Precision_8: 0.910
Recall_8: 0.793
F1_8: 0.847"""


import matplotlib.pyplot as plt

# Données fictives (à adapter selon ton code)
methods = ['Accuracy', 'Precision', 'Recall', 'F1-score']
cnn_rdft = [85.47, 84.79, 85.17, 84.84]
cnn_dwt = [78.48, 76.48, 76.34, 75.84]
cnn_mfcc = [85.61, 84.80, 84.85, 84.72]
cnn_combined = [83.74, 85.90, 81.61, 83.70]
vit1d = [Acc*100 , precision_avg*100, recall_avg*100, f1_avg*100]

plt.figure(figsize=(10, 6))

plt.plot(methods, cnn_rdft, marker='o', linestyle='--', label='CNN (RDFT)', linewidth=2)
plt.plot(methods, cnn_dwt, marker='o', linestyle='--', label='CNN (DWT)', linewidth=2)
plt.plot(methods, cnn_mfcc, marker='o', linestyle='--', label='CNN (MFCC)', linewidth=2)
plt.plot(methods, cnn_combined, marker='o', linestyle='--', label='CNN (RDFT+DWT+MFCC)', linewidth=2)
plt.plot(methods, vit1d, marker='o', linestyle='-', label='DASViT1D (proposed)', linewidth=2, color='brown')

plt.ylim(60, 100)
plt.ylabel('Percentage (%)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=11)
plt.grid(True)
plt.tight_layout()

plt.savefig("comparison_metrics.png", dpi=300)  # Export haute résolution
plt.show()


# import matplotlib.pyplot as plt

# # Données
# metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']

# # Valeurs pour chaque modèle (issus du tableau)
# cnn_rdft =     [85.47, 84.79, 85.17, 84.84]
# cnn_dwt =      [78.48, 76.48, 76.34, 75.84]
# cnn_mfcc =     [85.61, 84.80, 84.85, 84.72]
# gtn_combined = [93.5, 91.93, 90.8, 91.29]  # ViT1D avec RDFT+DWT+MFCC
# cnn_combined = [87.35, 85.57, 81.42, 83.19] 

# # Création du graphique
# plt.figure(figsize=(10, 6))

# plt.plot(metrics, cnn_rdft,     label='CNN (RDFT)',     linestyle='--', marker='o', color='#1F77B4')  # bleu foncé
# plt.plot(metrics, cnn_dwt,      label='CNN (DWT)',      linestyle='--', marker='o', color='#D62728')  # rouge foncé
# plt.plot(metrics, cnn_mfcc,     label='CNN (MFCC)',     linestyle='--', marker='o', color='#2CA02C')  # vert foncé
# plt.plot(metrics, cnn_combined, label='CNN (RDFT+DWT+MFCC)', linestyle='--', marker='o', color='#9467BD')  # violet foncé
# plt.plot(metrics, gtn_combined, label='DASViT1D (proposed)', linestyle='-', marker='o', color='#8C564B')  # brun foncé
# # Mise en forme
# plt.ylim(60, 100)
# plt.title('         ', fontsize=14)
# plt.ylabel('Percentage (%)', fontsize=12)
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.legend(fontsize=11)
# plt.tight_layout()

# # Affichage
# plt.show()

# from sklearn.base import BaseEstimator, ClassifierMixin
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# import numpy as np

# class ViT1DSklearnWrapper(BaseEstimator, ClassifierMixin):
#     def __init__(self, patch_size=512, emb_dim=128, num_heads=8, depth=6,
#                  batch_size=64, num_epochs=10, learning_rate=1e-3, input_dim=8192, num_classes=9, device=None):
#         self.patch_size = patch_size
#         self.emb_dim = emb_dim
#         self.num_heads = num_heads
#         self.depth = depth
#         self.batch_size = batch_size
#         self.num_epochs = num_epochs
#         self.learning_rate = learning_rate
#         self.input_dim = input_dim
#         self.num_classes = num_classes
#         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
#         self._build_model()

#     def _build_model(self):
#         self.model = ViT1D(
#             input_dim=self.input_dim,
#             patch_size=self.patch_size,
#             emb_dim=self.emb_dim,
#             num_heads=self.num_heads,
#             depth=self.depth,
#             num_classes=self.num_classes
#         ).to(self.device)

#     def fit(self, X, y):
#         X_tensor = torch.tensor(X, dtype=torch.float32)
#         y_tensor = torch.tensor(y, dtype=torch.long)
#         dataset = TensorDataset(X_tensor, y_tensor)
#         loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

#         self.model.train()
#         for epoch in range(self.num_epochs):
#             for xb, yb in loader:
#                 xb, yb = xb.to(self.device), yb.to(self.device)
#                 optimizer.zero_grad()
#                 preds = self.model(xb)
#                 loss = criterion(preds, yb)
#                 loss.backward()
#                 optimizer.step()

#         return self

#     def predict(self, X):
#         self.model.eval()
#         X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
#         with torch.no_grad():
#             preds = self.model(X_tensor)
#         return preds.argmax(dim=1).cpu().numpy()

#     def score(self, X, y):
#         preds = self.predict(X)
#         return np.mean(preds == y)


# from sklearn.model_selection import GridSearchCV

# param_grid = {
#     'patch_size': [256, 512],
#     'emb_dim': [64, 128],
#     'num_heads': [8, 16, 32],
#     'depth': [4, 8, 16],
#     'batch_size': [64,128,256],
#     'num_epochs': [100, 200]
# }

# model = ViT1DSklearnWrapper()

# grid = GridSearchCV(model, param_grid, scoring='accuracy', cv=3, verbose=2)
# grid.fit(X_train, y_train)

# print("Best parameters:", grid.best_params_)
# print("Best score:", grid.best_score_)

