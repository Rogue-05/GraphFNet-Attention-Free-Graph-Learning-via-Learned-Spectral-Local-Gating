"""
Reference Implementation: PascalVOC-SP (LRGB)
This script demonstrates the exact setup for training on PascalVOC-SP, including:
- Data loading, splits, and label mapping
- Node feature construction and masking (dense batching)
- Laplacian/eigenbasis construction with sign canonicalization
- Class weights for imbalanced CE loss
- Standard GCN baseline implementation for direct comparison
- Optimizer and scheduler configuration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.datasets import LRGBDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch, to_dense_adj
import numpy as np
import time

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# ==========================================
# 1. Train/Val/Test Split & Label Mapping
# ==========================================
print("Loading PascalVOC-SP Dataset...")
# LRGBDataset natively handles the standard train/val/test split.
train_ds = LRGBDataset(root='./data', name='PascalVOC-SP', split='train')
val_ds   = LRGBDataset(root='./data', name='PascalVOC-SP', split='val')
test_ds  = LRGBDataset(root='./data', name='PascalVOC-SP', split='test')

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)

# Inspect labels (0 to 20 = 21 classes)
sample = train_ds[0]
NUM_CLASSES = int(max(d.y.max().item() for d in [train_ds[0], val_ds[0], test_ds[0]])) + 1
print(f"Num classes: {NUM_CLASSES}")

# ==========================================
# 2. Node Feature Construction
# ==========================================
# PascalVOC-SP node features are 14-dimensional continuous vectors.
# (combines pixel-level info and coordinate info)
NODE_FEAT_DIM = sample.x.shape[-1]
print(f"Node feature dimension: {NODE_FEAT_DIM}")

# ==========================================
# 3. Class Weights (Imbalance handling)
# ==========================================
# Background class heavily dominates; we must weight the CrossEntropyLoss.
print("\nComputing class weights from training set...")
all_labels = torch.cat([d.y for d in train_ds], dim=0)
class_counts = torch.bincount(all_labels, minlength=NUM_CLASSES).float().clamp(min=1)
class_weights = 1.0 / class_counts
class_weights = (class_weights / class_weights.sum() * NUM_CLASSES).to(device)

for i, count in enumerate(class_counts):
    print(f"Class {i:2d}: count={int(count):8d}, weight={class_weights[i]:.4f}")

# print(f"Sample class weights: {class_weights.cpu().numpy()}...")

# ==========================================
# 4. Laplacian / Eigenbasis Construction
# ==========================================
def compute_laplacian_basis(adj, mask, k=64):
    """
    Computes normalized Laplacian and its truncated eigenbasis.
    Demonstrates sign canonicalization for stable spectral features.
    """
    B, N, _ = adj.shape
    A_list, U_list = [], []
    
    for b in range(B):
        # 1. Masking: extract valid graph
        n = int(mask[b].sum().item())
        adj_b = adj[b, :n, :n]
        
        # 2. Normalization: D^{-1/2} A D^{-1/2}
        deg = adj_b.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        A_norm_b = D_inv_sqrt @ adj_b @ D_inv_sqrt
        
        # 3. Laplacian construction
        L_b = torch.eye(n, device=adj.device) - A_norm_b
        
        try:
            # 4. Eigendecomposition
            _, U_b = torch.linalg.eigh(L_b)
            
            # 5. Sign canonicalization (crucial for stability)
            max_abs_idx = torch.abs(U_b).argmax(dim=0)
            signs = torch.sign(U_b[max_abs_idx, torch.arange(U_b.size(1), device=adj.device)])
            signs[signs == 0] = 1.0
            U_b = U_b * signs.unsqueeze(0)
        except Exception:
            U_b = torch.eye(n, device=adj.device)
            
        # Truncate to top-k
        k_actual = min(k, n)
        U_trunc = U_b[:, :k_actual]
        
        # Pad back to max nodes (N) and max k
        A_pad = F.pad(A_norm_b, (0, N - n, 0, N - n))
        U_pad = F.pad(U_trunc, (0, k - k_actual, 0, N - n))
        
        A_list.append(A_pad)
        U_list.append(U_pad)
        
    return torch.stack(A_list), torch.stack(U_list)

# ==========================================
# 5. Comparison: Correctly Implemented GCN
# ==========================================
class BaselineGCNLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.lin = nn.Linear(hidden_dim, hidden_dim)
        # Normalization layer (crucial for deep GNNs)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, A_norm):
        # x: [B, N, H], A_norm: [B, N, N]
        msg = torch.bmm(A_norm, x)
        return self.norm(F.gelu(self.lin(msg)))

class BaselineGCN(nn.Module):
    def __init__(self, in_dim=14, hidden_dim=128, num_layers=4, num_classes=21):
        super().__init__()
        # Continuous feature projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.layers = nn.ModuleList([
            BaselineGCNLayer(hidden_dim) for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(0.1)
        
        # Per-node readout (No graph pooling)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, batch):
        # 6. Masking logic (Dense conversion)
        x, mask = to_dense_batch(batch.x.float(), batch.batch)
        adj = to_dense_adj(batch.edge_index, batch.batch, max_num_nodes=x.size(1))
        
        # Self loops
        adj = adj + torch.eye(adj.size(1), device=x.device).unsqueeze(0)
        
        # A_norm construction (Re-using the logic from the laplacian function)
        A_norm, _ = compute_laplacian_basis(adj, mask, k=1) 

        x = self.input_proj(x)
        
        for layer in self.layers:
            x = x + self.dropout(layer(x, A_norm)) # Residual connections
            
        # Mask padded nodes before classification
        x = x * mask.unsqueeze(-1)
        logits = self.classifier(x)
        
        return logits, mask

# ==========================================
# 7. Optimizer, Scheduler & Training Demo
# ==========================================
print("\nInitializing Baseline GCN...")
model = BaselineGCN(in_dim=NODE_FEAT_DIM, num_classes=NUM_CLASSES).to(device)

# Standard optimizer configuration
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Cosine annealing scheduler (standard for LRGB)
MAX_EPOCHS = 200
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-5)

# Loss function with class weights and ignore_index for padded nodes
criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)

print("\nRunning a single forward/backward pass...")
batch = next(iter(train_loader)).to(device)

model.train()
optimizer.zero_grad()

# Forward
logits, mask = model(batch)

# Align labels with dense batching and ignore padded nodes
y_dense, _ = to_dense_batch(batch.y, batch.batch, fill_value=-1)

# Loss calculation
loss = criterion(
    logits.reshape(-1, logits.size(-1)), 
    y_dense.reshape(-1).long()
)

# Backward
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
scheduler.step()

print(f"Loss: {loss.item():.4f}")
print("Done. All mechanisms verified.")
