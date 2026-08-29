import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import average_precision_score
from torch_geometric.datasets import LRGBDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch, to_dense_adj
import os

class SimpleAtomEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # Peptides-func atom features: 9 categorical features
        self.embeddings = nn.ModuleList([
            nn.Embedding(64, hidden_dim),   # atomic num
            nn.Embedding(10, hidden_dim),   # chirality
            nn.Embedding(10, hidden_dim),   # degree
            nn.Embedding(10, hidden_dim),   # formal charge
            nn.Embedding(10, hidden_dim),   # num Hs
            nn.Embedding(10, hidden_dim),   # num radical e
            nn.Embedding(10, hidden_dim),   # hybridization
            nn.Embedding(3,  hidden_dim),   # aromaticity
            nn.Embedding(10, hidden_dim),   # ring membership
        ])
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x: [B, N, 9] integer features
        x = x.long().clamp(min=0)
        out = sum(emb(x[..., i]) for i, emb in enumerate(self.embeddings))
        return self.proj(F.gelu(out))
class DenseGCNLayer(nn.Module):
    def __init__(self, hidden_dim, edge_dim=3):
        super().__init__()
        self.node_lin = nn.Linear(hidden_dim, hidden_dim)
        # Project edge features to scalar weight [B,N,N] not [B,N,N,H]
        # Old approach created [B,N,N,H] intermediate = 2.6GB at batch=32
        self.edge_lin = nn.Linear(edge_dim, 1)
        self.norm     = nn.LayerNorm(hidden_dim)

    def forward(self, x, A_norm, edge_attr_dense=None):
        # x:               [B, N, H]
        # A_norm:          [B, N, N]
        # edge_attr_dense: [B, N, N, edge_dim]
        if edge_attr_dense is not None:
            # Scalar edge gate [B, N, N] — same memory as adj, not H times more
            E   = torch.sigmoid(self.edge_lin(edge_attr_dense)).squeeze(-1)  # [B,N,N]
            msg = torch.bmm(A_norm * E, x)                                   # [B,N,H]
        else:
            msg = torch.bmm(A_norm, x)

        return self.norm(F.gelu(self.node_lin(msg)))
class SpectralMixMH(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.num_heads   = num_heads
        self.head_dim    = hidden_dim // num_heads
        self.filter_gen  = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj    = nn.Linear(hidden_dim, hidden_dim)
        self.norm        = nn.LayerNorm(hidden_dim)

    def forward(self, x, U, mask):
        # x: [B, N, H],  U: [B, N, N]
        B, N, H = x.shape

        # Spectral domain: x_hat = U^T x
        x_hat     = torch.bmm(U.transpose(1, 2), x)         # [B, N, H]

        # Learned per-node spectral filter
        fil       = torch.sigmoid(self.filter_gen(x_hat))   # [B, N, H]
        x_filtered = fil * x_hat                             # [B, N, H]

        # Back to spatial: x_out = U x_filtered
        x_out = torch.bmm(U, x_filtered)                    # [B, N, H]

        # Zero out padded positions
        x_out = x_out * mask.unsqueeze(-1)

        return self.norm(self.out_proj(F.gelu(x_out)))
class GatedPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, mask):
        # x:    [B, N, H]
        # mask: [B, N]  bool
        scores  = self.gate(x).squeeze(-1)                   # [B, N]
        scores  = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1) # [B, N, 1]
        return (x * weights).sum(dim=1)                      # [B, H]
class HybridGraphFNet_Best(nn.Module):
    def __init__(
        self,
        hidden_dim  = 128,
        num_layers  = 4,
        out_dim     = 10,
        num_heads   = 4,
        lap_k       = 8,
        dropout     = 0.1,
        edge_dim    = 3,
    ):
        super().__init__()
        self.lap_k   = lap_k
        self.dropout = nn.Dropout(dropout)

        # --- Encoders ---
        self.input_proj  = SimpleAtomEncoder(hidden_dim)
        self.pe_encoder  = nn.Linear(lap_k, hidden_dim)

        # --- Layers ---
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "local":  DenseGCNLayer(hidden_dim, edge_dim=edge_dim),
                "global": SpectralMixMH(hidden_dim, num_heads=num_heads),
                "gate":   nn.Linear(hidden_dim, hidden_dim),
                "norm":   nn.LayerNorm(hidden_dim),
            })
            for _ in range(num_layers)
        ])

        # --- Readout ---
        self.pool = GatedPooling(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    # ------------------------------------------------------------------
    def compute_laplacian_basis(self, adj, mask):
      B, N, _ = adj.shape
      A_list, U_list = [], []

      for b in range(B):
          n = int(mask[b].sum().item())
          adj_b = adj[b, :n, :n]

          deg          = adj_b.sum(dim=1)
          deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)
          D_inv_sqrt   = torch.diag(deg_inv_sqrt)
          A_norm_b     = D_inv_sqrt @ adj_b @ D_inv_sqrt

          L_b = torch.eye(n, device=adj.device) - A_norm_b

          try:
              _, U_b = torch.linalg.eigh(L_b)

              # ---- Sign fix ----
              # For each eigenvector column, find the element with largest
              # absolute value and force it to be positive
              max_abs_idx = torch.abs(U_b).argmax(dim=0)          # [n]
              signs = torch.sign(
                  U_b[max_abs_idx, torch.arange(U_b.size(1), device=adj.device)]
              )                                                     # [n]
              signs[signs == 0] = 1.0                              # avoid multiply by 0
              U_b = U_b * signs.unsqueeze(0)                       # [n, n]

          except Exception:
              U_b = torch.eye(n, device=adj.device)

          # Pad back to N
          A_pad = F.pad(A_norm_b, (0, N - n, 0, N - n))
          U_pad = F.pad(U_b,      (0, N - n, 0, N - n))
          A_list.append(A_pad)
          U_list.append(U_pad)

      return torch.stack(A_list), torch.stack(U_list)

    # ------------------------------------------------------------------
    def forward(self, data):
        # ---- Dense conversion ----
        x,    mask = to_dense_batch(data.x.float(), data.batch)  # [B,N,F], [B,N]
        adj         = to_dense_adj(
            data.edge_index, data.batch,
            max_num_nodes=x.size(1)
        )                                                         # [B,N,N]

        # Edge attributes (3 features: bond type, stereo, is_aromatic)
        if data.edge_attr is not None:
            edge_attr_dense = to_dense_adj(
                data.edge_index, data.batch,
                edge_attr=data.edge_attr[:, :3].float(),
                max_num_nodes=x.size(1)
            )                                                     # [B,N,N,3]
        else:
            edge_attr_dense = None

        # Self-loops
        I   = torch.eye(adj.size(1), device=x.device).unsqueeze(0)
        adj = adj + I

        # ---- Spectral basis (clean, per-graph) ----
        A_norm, U = self.compute_laplacian_basis(adj, mask)

        # ---- Node features ----
        x = self.input_proj(x)                                   # [B,N,H]

        # ---- Laplacian PE injection ----
        k       = min(self.lap_k, U.size(-1))
        lap_pe  = U[:, :, :k]                                    # [B,N,k]
        lap_pe  = lap_pe * mask.unsqueeze(-1)
        x       = x + self.pe_encoder(lap_pe)                    # [B,N,H]

        # ---- Message passing ----
        for layer in self.layers:
            x_res    = x
            x_local  = layer["local"](x, A_norm, edge_attr_dense)
            x_global = layer["global"](x, U, mask)

            gate  = torch.sigmoid(layer["gate"](x))
            x_mix = gate * x_local + (1 - gate) * x_global

            x = layer["norm"](x_res + self.dropout(x_mix))

        # ---- Readout ----
        x = x * mask.unsqueeze(-1)
        graph_emb = self.pool(x, mask)                           # [B,H]

        return self.classifier(graph_emb)
    
class HybridGraphFNet_Best_Peptides(HybridGraphFNet_Best):
    def __init__(self, hidden_dim=128, num_layers=4, out_dim=10):
        super().__init__(
            hidden_dim = hidden_dim,
            num_layers = num_layers,
            out_dim    = out_dim,
            num_heads  = 4,
            lap_k      = 8,
            dropout    = 0.1,
            edge_dim   = 3,
        )
        # input_proj already set to SimpleAtomEncoder in parent
