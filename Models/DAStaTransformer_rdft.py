
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






class ViT1D(nn.Module):
    def __init__(self, input_dim=2580, patch_size=43, emb_dim=64, num_heads=8, depth=8, num_classes=9):
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
        x = x.mean(dim=1)
        output = self.cls_head(x)
        
        return output
    
   


import pickle

# Charger les splits
with open("splits/split_indices.pkl", "rb") as f:
    split_dict = pickle.load(f)

train_idx = split_dict['train_idx']
val_idx = split_dict['val_idx']
test_idx = split_dict['test_idx']

# Charger ton dataset spécifique (ex: MFCC ou RDFT)
data_total = pd.read_csv("rdft_feat_data.csv", header=None)
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

# model = create_model()
# model.fit(X_train, y_train, ...)  # ton entraînement





# X = data_total.iloc[:, :-1].values   
# y = data_total.iloc[:, -1].values   


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)



scaler = StandardScaler().fit(X_train)

Xtrain = scaler.transform(X_train)
Xval = scaler.transform(X_val)
Xtest = scaler.transform(X_test)


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
num_epochs = 100
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


C = confusion_matrix(labels, predictions)
print('Confusion Matrix:\n', C)
print('Training time:', end_train - start_train)
print('Testing time:', end_test - start_test)

# Heatmap

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
df = pd.DataFrame(C)

sns.heatmap(df, fmt='g', annot=True, robust=True,
            annot_kws={'size': 12},
            xticklabels=['Car','Fence', 'Longboard', 'Manip', 'Manhole', 'Regular', 'Construction','Running','Walking'],
            yticklabels=['Car','Fence', 'Longboard', 'Manip', 'Manhole', 'Regular', 'Construction','Running','Walking'],
            cmap='Reds',  # plus proche de l'image que 'Reds'
            linewidths=0.5, linecolor='white', square=True, cbar=True)

# Ajustements axes
ax.set_xlabel('Predicted label', fontsize=14)
ax.set_ylabel('True label', fontsize=14)

# Rotation X ticks
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(rotation=0, fontsize=12)

# Colorbar style
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=12)

# Marges plus serrées
plt.tight_layout()

# Sauvegarde
plt.savefig('confusion_matrix_paper_style.png', dpi=300)
plt.show()

# Métriques
Acc = np.trace(C) / np.sum(C)
print('Accuracy: %.4f' % Acc)
NAR = (np.sum(C[5]) - C[5][5]) / np.sum(C[:, 1:])
print('NAR: %.4f' % NAR)
FNR = (np.sum(C[:, 5]) - C[5][5]) / np.sum(C[1:, :])
print('FNR: %.4f' % FNR)

column_sum = np.sum(C, axis=0)
row_sum = np.sum(C, axis=1)
print('Column sums:', column_sum)
print('Row sums:', row_sum)

for i in range(0, 8):
    Precision = C[i-1, i-1] / column_sum[i-1] if column_sum[i-1] != 0 else 0
    Recall = C[i-1, i-1] / row_sum[i-1] if row_sum[i-1] != 0 else 0
    F1 = 2 * Precision * Recall / (Precision + Recall) if (Precision + Recall) != 0 else 0
    print(f'Precision_{i}: {Precision:.3f}')
    print(f'Recall_{i}: {Recall:.3f}')
    print(f'F1_{i}: {F1:.3f}')

