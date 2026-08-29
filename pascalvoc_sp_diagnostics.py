"""
PascalVOC-SP Per-Class Diagnostic & Majority Baseline Script (Fixed)

Calculates:
1. Trivial Majority-Class Baseline (Macro F1 & Per-Class F1) on Val & Test sets.
2. Per-Class F1 breakdown for trained HybridGraphFNet checkpoints (best_model_voc_sp_seedX.pt / .zip / folder in models/).
3. Diagnostic analysis: Checks if Macro F1 is near the trivial baseline and whether per-class scores confirm class imbalance issues.
"""

import os
import io
import time
import zipfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch_geometric.datasets import LRGBDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch, to_dense_adj

# PascalVOC-SP Class Names (21 classes)
CLASS_NAMES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# ==============================================================================
# 1. Dataset & Paths Setup
# ==============================================================================
DATA_ROOT = './data' if os.path.exists('./data') else '../data' if os.path.exists('../data') else './new/data' if os.path.exists('./new/data') else './data'

print(f"Loading datasets from '{DATA_ROOT}'...")
train_ds = LRGBDataset(root=DATA_ROOT, name='PascalVOC-SP', split='train')
val_ds   = LRGBDataset(root=DATA_ROOT, name='PascalVOC-SP', split='val')
test_ds  = LRGBDataset(root=DATA_ROOT, name='PascalVOC-SP', split='test')

NUM_CLASSES = 21
NODE_FEAT_DIM = train_ds[0].x.shape[-1]
sample_edge = train_ds[0].edge_attr
EDGE_DIM = sample_edge.shape[-1] if (sample_edge is not None and sample_edge.dim() > 1) else 1

print(f"Dataset stats: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

# ==============================================================================
# 2. Eigenbasis Cache Setup + Precomputation (if missing)
# ==============================================================================
possible_cache_dirs = ['./eigenbasis_cache_voc_sp', '../eigenbasis_cache_voc_sp', './new/eigenbasis_cache_voc_sp']
CACHE_DIR = './eigenbasis_cache_voc_sp'
for cdir in possible_cache_dirs:
    if os.path.exists(cdir):
        CACHE_DIR = cdir
        break

TRUNC_K = 64
print(f"Eigenbasis cache directory: {CACHE_DIR}")

def precompute_eigenbasis_for_split(dataset, split_name, cache_dir, k=64):
    split_dir = os.path.join(cache_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    num_graphs = len(dataset)
    eigh_failures = 0
    total_time = 0.0
    total_bytes = 0

    pbar = tqdm(range(num_graphs), desc=f'Precomputing eigenbasis [{split_name}]')
    for idx in pbar:
        data = dataset[idx]
        n = data.num_nodes

        edge_index = data.edge_index
        adj = torch.zeros(n, n)
        adj[edge_index[0], edge_index[1]] = 1.0
        adj = adj + torch.eye(n)

        deg = adj.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        A_norm = D_inv_sqrt @ adj @ D_inv_sqrt
        L = torch.eye(n) - A_norm

        t0 = time.perf_counter()
        try:
            _, U = torch.linalg.eigh(L)
            max_abs_idx = torch.abs(U).argmax(dim=0)
            signs = torch.sign(U[max_abs_idx, torch.arange(U.size(1))])
            signs[signs == 0] = 1.0
            U = U * signs.unsqueeze(0)
        except Exception as e:
            eigh_failures += 1
            U = torch.eye(n)

        total_time += time.perf_counter() - t0
        k_actual = min(k, n)
        U_trunc = U[:, :k_actual]

        save_path = os.path.join(split_dir, f'{idx}.pt')
        torch.save({'U': U_trunc.clone(), 'n': n, 'k': k_actual}, save_path)
        total_bytes += os.path.getsize(save_path)

    print(f"  {split_name} cache generated: {num_graphs} graphs, {total_bytes/1e6:.1f} MB")

# Verify / build cache if missing
for split_name, ds in [('train', train_ds), ('val', val_ds), ('test', test_ds)]:
    split_dir = os.path.join(CACHE_DIR, split_name)
    expected = len(ds)
    existing = len([f for f in os.listdir(split_dir) if f.endswith('.pt')]) if os.path.isdir(split_dir) else 0

    if existing < expected:
        print(f"Generating missing eigenbasis cache for {split_name} ({existing}/{expected} cached)...")
        precompute_eigenbasis_for_split(ds, split_name, CACHE_DIR, k=TRUNC_K)
    else:
        print(f"{split_name} eigenbasis cache verified ({existing}/{expected} files).")

# ==============================================================================
# 3. Cached Dataset Wrapper & DataLoaders
# ==============================================================================
class CachedEigenbasisDataset:
    def __init__(self, base_dataset, cache_dir, split_name, k_trunc=64):
        self.base = base_dataset
        self.split_dir = os.path.join(cache_dir, split_name)
        self.k_trunc = k_trunc

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        data = self.base[idx].clone()
        cache_path = os.path.join(self.split_dir, f'{idx}.pt')
        
        if os.path.exists(cache_path):
            cache = torch.load(cache_path, weights_only=True)
            U = cache['U']
            if U.size(1) < self.k_trunc:
                U = F.pad(U, (0, self.k_trunc - U.size(1)))
            data.cached_U = U
        else:
            n = data.num_nodes
            data.cached_U = torch.eye(n)[:, :min(n, self.k_trunc)]
            if data.cached_U.size(1) < self.k_trunc:
                data.cached_U = F.pad(data.cached_U, (0, self.k_trunc - data.cached_U.size(1)))

        return data

train_loader = DataLoader(CachedEigenbasisDataset(train_ds, CACHE_DIR, 'train', k_trunc=TRUNC_K), batch_size=4, shuffle=False)
val_loader   = DataLoader(CachedEigenbasisDataset(val_ds, CACHE_DIR, 'val', k_trunc=TRUNC_K), batch_size=4, shuffle=False)
test_loader  = DataLoader(CachedEigenbasisDataset(test_ds, CACHE_DIR, 'test', k_trunc=TRUNC_K), batch_size=4, shuffle=False)

# ==============================================================================
# 4. Checkpoint Loader Function
# ==============================================================================
def load_checkpoint_dict(ckpt_path, target_device):
    """Loads state dict from file (.pt, .zip) or unzipped PyTorch archive folder."""
    if os.path.isdir(ckpt_path):
        buffer = io.BytesIO()
        prefix = os.path.basename(ckpt_path)
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_STORED) as zf:
            for root, dirs, files in os.walk(ckpt_path):
                for f in files:
                    if f.startswith('.'): continue
                    full_p = os.path.join(root, f)
                    rel_p = os.path.join(prefix, os.path.relpath(full_p, ckpt_path))
                    zinfo = zipfile.ZipInfo(filename=rel_p, date_time=(2024, 1, 1, 0, 0, 0))
                    with open(full_p, 'rb') as fp:
                        zf.writestr(zinfo, fp.read())
        buffer.seek(0)
        state = torch.load(buffer, map_location=target_device, weights_only=True)
    else:
        try:
            state = torch.load(ckpt_path, map_location=target_device, weights_only=True)
        except Exception:
            state = torch.load(ckpt_path, map_location=target_device, weights_only=False)
            
    if isinstance(state, dict) and 'model_state' in state:
        state = state['model_state']
    return state

# ==============================================================================
# 5. Model Architecture
# ==============================================================================
class DenseGCNLayer(nn.Module):
    def __init__(self, hidden_dim, edge_dim=1):
        super().__init__()
        self.node_lin = nn.Linear(hidden_dim, hidden_dim)
        self.edge_lin = nn.Linear(edge_dim, 1)
        self.norm     = nn.LayerNorm(hidden_dim)

    def forward(self, x, A_norm, edge_attr_dense=None):
        if edge_attr_dense is not None:
            E   = torch.sigmoid(self.edge_lin(edge_attr_dense)).squeeze(-1)
            msg = torch.bmm(A_norm * E, x)
        else:
            msg = torch.bmm(A_norm, x)
        return self.norm(F.gelu(self.node_lin(msg)))

class SpectralMixMH(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.num_heads  = num_heads
        self.head_dim   = hidden_dim // num_heads
        self.filter_gen = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj   = nn.Linear(hidden_dim, hidden_dim)
        self.norm       = nn.LayerNorm(hidden_dim)

    def forward(self, x, U, mask):
        x_hat      = torch.bmm(U.transpose(1, 2), x)
        fil        = torch.sigmoid(self.filter_gen(x_hat))
        x_filtered = fil * x_hat
        x_out      = torch.bmm(U, x_filtered)
        x_out      = x_out * mask.unsqueeze(-1)
        return self.norm(self.out_proj(F.gelu(x_out)))

class HybridGraphFNet_NodeLevel(nn.Module):
    def __init__(self, in_dim=14, hidden_dim=128, num_layers=4, num_classes=21,
                 num_heads=4, lap_k=8, dropout=0.1, edge_dim=1):
        super().__init__()
        self.lap_k   = lap_k
        self.dropout = nn.Dropout(dropout)
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.pe_encoder = nn.Linear(lap_k, hidden_dim)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'local':  DenseGCNLayer(hidden_dim, edge_dim=edge_dim),
                'global': SpectralMixMH(hidden_dim, num_heads=num_heads),
                'gate':   nn.Linear(hidden_dim, hidden_dim),
                'norm':   nn.LayerNorm(hidden_dim),
            }) for _ in range(num_layers)
        ])

        self.node_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes)
        )

    def compute_A_norm(self, adj, mask):
        B, N, _ = adj.shape
        A_list = []
        for b in range(B):
            n = int(mask[b].sum().item())
            adj_b = adj[b, :n, :n]
            deg = adj_b.sum(dim=1)
            deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)
            D_inv_sqrt = torch.diag(deg_inv_sqrt)
            A_norm_b = D_inv_sqrt @ adj_b @ D_inv_sqrt
            A_list.append(F.pad(A_norm_b, (0, N-n, 0, N-n)))
        return torch.stack(A_list)

    def forward(self, data):
        x, mask = to_dense_batch(data.x.float(), data.batch)
        adj = to_dense_adj(data.edge_index, data.batch, max_num_nodes=x.size(1))
        if data.edge_attr is not None:
            ea = data.edge_attr.float()
            if ea.dim() == 1: ea = ea.unsqueeze(-1)
            edge_attr_dense = to_dense_adj(data.edge_index, data.batch, edge_attr=ea, max_num_nodes=x.size(1))
        else:
            edge_attr_dense = None

        adj = adj + torch.eye(adj.size(1), device=x.device).unsqueeze(0)
        A_norm = self.compute_A_norm(adj, mask)

        U, _ = to_dense_batch(data.cached_U.float(), data.batch)
        U = U * mask.unsqueeze(-1)

        x = self.input_proj(x)
        k = min(self.lap_k, U.size(-1))
        lap_pe = U[:, :, :k] * mask.unsqueeze(-1)
        x = x + self.pe_encoder(lap_pe)

        for layer in self.layers:
            x_res    = x
            x_local  = layer['local'](x, A_norm, edge_attr_dense)
            x_global = layer['global'](x, U, mask)
            gate     = torch.sigmoid(layer['gate'](x))
            x_mix    = gate * x_local + (1 - gate) * x_global
            x        = layer['norm'](x_res + self.dropout(x_mix))

        x = x * mask.unsqueeze(-1)
        logits = self.node_classifier(x)
        return logits, mask

# ==============================================================================
# 6. Trivial Majority Baseline Calculation
# ==============================================================================
print("\n" + "="*80)
print("1. TRIVIAL MAJORITY-CLASS BASELINE")
print("="*80)

train_labels = torch.cat([d.y for d in train_ds], dim=0).numpy()
class_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
majority_class = int(np.argmax(class_counts))

print(f"Majority Class in Training Set: Class {majority_class} ('{CLASS_NAMES[majority_class]}')")
print(f"  Count: {class_counts[majority_class]:,} / {len(train_labels):,} nodes ({100.0 * class_counts[majority_class] / len(train_labels):.2f}%)")

def evaluate_majority_baseline(dataset, majority_cls):
    all_labels = torch.cat([d.y for d in dataset], dim=0).numpy()
    preds = np.full_like(all_labels, fill_value=majority_cls)
    
    macro_f1 = f1_score(all_labels, preds, average='macro', zero_division=0)
    per_class_f1 = f1_score(all_labels, preds, average=None, zero_division=0)
    node_counts = np.bincount(all_labels, minlength=NUM_CLASSES)
    return macro_f1, per_class_f1, node_counts

val_maj_macro, val_maj_per_class, val_counts = evaluate_majority_baseline(val_ds, majority_class)
test_maj_macro, test_maj_per_class, test_counts = evaluate_majority_baseline(test_ds, majority_class)

print(f"\nMajority Baseline Macro F1:")
print(f"  Validation Set: {val_maj_macro:.4f}")
print(f"  Test Set:       {test_maj_macro:.4f}  (Background F1 = {test_maj_per_class[0]:.4f}, all other classes = 0.0000)")

# ==============================================================================
# 7. Evaluate Trained Checkpoints
# ==============================================================================
print("\n" + "="*80)
print("2. EVALUATING TRAINED CHECKPOINTS IN models/")
print("="*80)

model_dirs = ['./models', '../models', '.', './new']
ckpt_files = []
for d in model_dirs:
    if os.path.exists(d):
        for f in os.listdir(d):
            if 'best_model_voc_sp_seed' in f:
                ckpt_files.append(os.path.join(d, f))

ckpt_files = sorted(list(set(ckpt_files)))
print(f"Found {len(ckpt_files)} checkpoint files/directories: {[os.path.basename(f) for f in ckpt_files]}")

def evaluate_model_per_class(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            batch = batch.to(device)
            logits, mask = model(batch)
            y_dense, _ = to_dense_batch(batch.y, batch.batch)
            valid = mask.bool()
            all_preds.append(logits[valid].argmax(dim=-1).cpu())
            all_labels.append(y_dense[valid].cpu())
            
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    return macro_f1, per_class_f1, all_preds, all_labels

seed_results = {}
for ckpt_path in ckpt_files:
    filename = os.path.basename(ckpt_path)
    seed_str = filename.split('seed')[-1].split('.')[0].replace('.pt','').replace('.zip','')
    seed = int(seed_str) if seed_str.isdigit() else filename
    
    print(f"\nLoading checkpoint: {filename}...")
    model = HybridGraphFNet_NodeLevel(
        in_dim=NODE_FEAT_DIM, hidden_dim=128, num_layers=4,
        num_classes=NUM_CLASSES, num_heads=4, lap_k=8, edge_dim=EDGE_DIM
    ).to(device)
    
    state_dict = load_checkpoint_dict(ckpt_path, device)
    model.load_state_dict(state_dict)
        
    val_macro, val_per_class, _, _ = evaluate_model_per_class(model, val_loader)
    test_macro, test_per_class, test_preds, test_labels = evaluate_model_per_class(model, test_loader)
    
    seed_results[seed] = {
        'val_macro': val_macro, 'test_macro': test_macro,
        'val_per_class': val_per_class, 'test_per_class': test_per_class,
        'test_preds': test_preds, 'test_labels': test_labels
    }
    print(f"  Seed {seed} Macro F1 -> Val: {val_macro:.4f} | Test: {test_macro:.4f}")

# ==============================================================================
# 8. Detailed Per-Class Comparison Table
# ==============================================================================
print("\n" + "="*80)
print("3. PER-CLASS F1 BREAKDOWN TABLE (TEST SET)")
print("="*80)

total_test_nodes = sum(test_counts)
seeds = sorted(list(seed_results.keys()))

header = f"{'ID':<3} {'Class Name':<14} {'Support':<10} {'% Nodes':<8} {'Maj Baseline':<12}"
for s in seeds:
    header += f" {'Seed '+str(s):<10}"
if len(seeds) > 1:
    header += f" {'Mean F1':<10}"
print(header)
print("-" * len(header))

for c in range(NUM_CLASSES):
    count = test_counts[c]
    pct = 100.0 * count / total_test_nodes
    maj_f1 = test_maj_per_class[c]
    
    row = f"{c:<3d} {CLASS_NAMES[c]:<14} {count:<10,d} {pct:<7.2f}% {maj_f1:<12.4f}"
    
    c_f1s = []
    for s in seeds:
        f1_s = seed_results[s]['test_per_class'][c]
        c_f1s.append(f1_s)
        row += f" {f1_s:<10.4f}"
        
    if len(seeds) > 1:
        mean_c_f1 = np.mean(c_f1s)
        row += f" {mean_c_f1:<10.4f}"
        
    print(row)

print("-" * len(header))
summary_row = f"{'MACRO F1 AVERAGE':<28} {total_test_nodes:<10,d} 100.00% {test_maj_macro:<12.4f}"
for s in seeds:
    summary_row += f" {seed_results[s]['test_macro']:<10.4f}"
if len(seeds) > 1:
    all_macros = [seed_results[s]['test_macro'] for s in seeds]
    summary_row += f" {np.mean(all_macros):<10.4f}"
print(summary_row)
print("="*80)

# ==============================================================================
# 9. Diagnostic Conclusions
# ==============================================================================
print("\n" + "="*80)
print("4. DIAGNOSTIC CONCLUSIONS")
print("="*80)

if seeds:
    primary_seed = seeds[0]
    m_macro = seed_results[primary_seed]['test_macro']
    m_per_class = seed_results[primary_seed]['test_per_class']
    
    diff_from_maj = m_macro - test_maj_macro
    print(f"1. MACRO F1 VS MAJORITY BASELINE:")
    print(f"   Model Test Macro F1:      {m_macro:.4f}")
    print(f"   Trivial Majority F1:      {test_maj_macro:.4f}")
    print(f"   Difference:               {diff_from_maj:+.4f}")
    
    bg_f1 = m_per_class[0]
    non_bg_f1s = m_per_class[1:]
    near_zero_classes = sum(1 for f in non_bg_f1s if f < 0.01)
    
    print(f"\n2. CLASS F1 DISTRIBUTION:")
    print(f"   Background Class (0) F1:  {bg_f1:.4f} (Support: {test_counts[0]:,} nodes, {100.0*test_counts[0]/total_test_nodes:.1f}%)")
    print(f"   Non-Background Mean F1:   {np.mean(non_bg_f1s):.4f}")
    print(f"   Classes near-zero F1 (<0.01): {near_zero_classes} / 20 non-background classes")

print("\nDiagnostics complete.")
