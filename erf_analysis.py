import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
from torch_geometric.utils import to_dense_batch, to_dense_adj, to_networkx
from torch_geometric.data import Batch
from torch_geometric.datasets import LRGBDataset

from model import HybridGraphFNet_Best_Peptides  # <-- Make sure this import matches your actual model file and class name

# TODO: Import your model definition here
# from your_model_file import HybridGraphFNet_Best_Peptides

def compute_erf(model, batch, target_node_idx, device):
    model.eval()

    x, mask = to_dense_batch(batch.x.float(), batch.batch)
    adj = to_dense_adj(batch.edge_index, batch.batch, max_num_nodes=x.size(1))
    I = torch.eye(adj.size(1), device=device).unsqueeze(0)
    adj = adj + I

    if batch.edge_attr is not None:
        edge_attr_dense = to_dense_adj(
            batch.edge_index, batch.batch,
            edge_attr=batch.edge_attr[:, :3].float(),
            max_num_nodes=x.size(1)
        )
    else:
        edge_attr_dense = None

    # --- THE FIX: Safely compute Laplacian Basis on CPU to avoid MPS collapse ---
    B, N, _ = adj.shape
    A_list, U_list = [], []
    for b in range(B):
        n = int(mask[b].sum().item())
        adj_b = adj[b, :n, :n]
        deg = adj_b.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        A_norm_b = D_inv_sqrt @ adj_b @ D_inv_sqrt
        
        L_b = torch.eye(n, device=device) - A_norm_b
        
        # Force CPU computation
        L_b_cpu = L_b.cpu()
        try:
            _, U_b_cpu = torch.linalg.eigh(L_b_cpu)
            U_b = U_b_cpu.to(device)
            
            # Sign fix
            max_abs_idx = torch.abs(U_b).argmax(dim=0)
            signs = torch.sign(U_b[max_abs_idx, torch.arange(U_b.size(1), device=device)])
            signs[signs == 0] = 1.0
            U_b = U_b * signs.unsqueeze(0)
        except Exception as e:
            print(f"Warning: Eigh failed on CPU: {e}")
            U_b = torch.eye(n, device=device)
            
        A_pad = F.pad(A_norm_b, (0, N - n, 0, N - n))
        U_pad = F.pad(U_b,      (0, N - n, 0, N - n))
        A_list.append(A_pad)
        U_list.append(U_pad)
        
    A_norm = torch.stack(A_list)
    U = torch.stack(U_list)
    # --------------------------------------------------------------------------

    # 1. Project raw discrete features into continuous embeddings
    x_enc = model.input_proj(x)

    # 2. Attach requires_grad to measure influence
    x_enc = x_enc.detach().requires_grad_(True)

    k = min(model.lap_k, U.size(-1))
    lap_pe = U[:, :, :k] * mask.unsqueeze(-1)
    if k < model.lap_k:
        lap_pe = F.pad(lap_pe, (0, model.lap_k - k))

    # 3. Create a new variable `h`
    h = x_enc + model.pe_encoder(lap_pe)

    for layer in model.layers:
        x_res = h
        x_local = layer["local"](h, A_norm, edge_attr_dense)
        x_global = layer["global"](h, U, mask)
        gate = torch.sigmoid(layer["gate"](h))
        x_mix = gate * x_local + (1 - gate) * x_global
        h = layer["norm"](x_res + model.dropout(x_mix))

    target_out = h[0, target_node_idx, :].sum()
    target_out.backward()

    influence = x_enc.grad[0, :, :].norm(dim=-1)     
    influence = influence * mask[0]                  
    
    max_inf = influence.max()
    if max_inf > 0:
        influence = influence / max_inf
        
    return influence.detach().cpu()


def aggregate_hop_influence(data, influence, target_node_idx, hop_stats):
    """Calculates graph distances and appends influence scores to the stats dictionary."""
    G = to_networkx(data, to_undirected=True)
    lengths = dict(nx.single_source_shortest_path_length(G, target_node_idx))
    
    influence_np = influence[:data.num_nodes].numpy()
    
    for node, hop in lengths.items():
        hop_stats[hop].append(influence_np[node])
    return hop_stats


def print_and_plot_stats(hop_stats_struct, hop_stats_func, save_path="erf_comparison.png"):
    """Generates the statistical table and error-bar plot."""
    # Find max hop to align both dictionaries
    max_hop = max(max(hop_stats_struct.keys(), default=0), max(hop_stats_func.keys(), default=0))
    
    hops = list(range(max_hop + 1))
    struct_means, struct_stds = [], []
    func_means, func_stds = [], []

    print("\n" + "="*60)
    print(f"{'Hop':<5} | {'Peptides-struct (Mean ± SD)':<25} | {'Peptides-func (Mean ± SD)':<25}")
    print("-" * 60)

    for hop in hops:
        # Struct stats
        s_vals = hop_stats_struct.get(hop, [])
        s_mean = np.mean(s_vals) if s_vals else 0.0
        s_std = np.std(s_vals) if s_vals else 0.0
        struct_means.append(s_mean)
        struct_stds.append(s_std)

        # Func stats
        f_vals = hop_stats_func.get(hop, [])
        f_mean = np.mean(f_vals) if f_vals else 0.0
        f_std = np.std(f_vals) if f_vals else 0.0
        func_means.append(f_mean)
        func_stds.append(f_std)

        print(f"{hop:<5} | {s_mean:.4f} ± {s_std:.4f}{'':<8} | {f_mean:.4f} ± {f_std:.4f}")
        
    print("="*60 + "\n")

    # Plotting
    x = np.arange(len(hops))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, struct_means, width, yerr=struct_stds, label='Struct', color='skyblue', capsize=5)
    ax.bar(x + width/2, func_means, width, yerr=func_stds, label='Func', color='salmon', capsize=5)

    ax.set_xlabel('Hop Distance from Target Node')
    ax.set_ylabel('Normalized Influence Score')
    ax.set_title('Aggregated ERF over 50 Graphs: Struct vs Func')
    ax.set_xticks(x)
    ax.set_xticklabels(hops)
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Chart saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    # Device configuration
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Load dataset
    print("Loading LRGB Peptides dataset...")
    test_data = LRGBDataset(root="data", name="Peptides-struct", split="test")

    # Sample 50 random graphs
    num_samples = 50
    sample_indices = random.sample(range(len(test_data)), num_samples)
    print(f"Selected {num_samples} random graphs for evaluation.")

    # Define model folder
    MODEL_DIR = "models" # <-- Update this if your folder name is different
    
    # Initialize models
    print("Loading models...")
    model_struct = HybridGraphFNet_Best_Peptides(hidden_dim=128, num_layers=4, out_dim=11)
    model_struct.load_state_dict(
        torch.load(os.path.join(MODEL_DIR, "best_model_struct_Best_Peptides_seed0.pt"), map_location=device)
    )
    model_struct = model_struct.to(device)

    model_func = HybridGraphFNet_Best_Peptides(hidden_dim=128, num_layers=4, out_dim=10)
    model_func.load_state_dict(
        torch.load(os.path.join(MODEL_DIR, "best_model_func_Best_Peptides_seed1.pt"), map_location=device),
        strict=False
    )
    model_func = model_func.to(device)

    # Data structures to hold aggregated stats (hop_distance -> list of influences)
    hop_stats_struct = defaultdict(list)
    hop_stats_func = defaultdict(list)

    print("Computing ERF...")
    for idx, data_idx in enumerate(sample_indices):
        sample = test_data[data_idx]
        batch = Batch.from_data_list([sample]).to(device)
        
        # Pick a target node (e.g., the center-most node index)
        target = sample.num_nodes // 2
        
        # Compute Struct Influence
        inf_struct = compute_erf(model_struct, batch, target, device)
        hop_stats_struct = aggregate_hop_influence(sample, inf_struct, target, hop_stats_struct)

        # Compute Func Influence
        inf_func = compute_erf(model_func, batch, target, device)
        hop_stats_func = aggregate_hop_influence(sample, inf_func, target, hop_stats_func)

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{num_samples} graphs...")

    # Print table and save plot
    print_and_plot_stats(hop_stats_struct, hop_stats_func, save_path="aggregated_erf_comparison.png")