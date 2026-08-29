# HybridGraphFNet — Full Project Context Dump
### For teammates writing the paper

---

## 1. Project Summary

We propose **HybridGraphFNet**, a novel graph neural network that replaces self-attention with learned spectral mixing via the graph Laplacian eigenbasis, gated against local GCN aggregation on a per-node, per-feature basis. The architecture is inspired by FNet (Lee-Thorp et al., 2021), which showed that attention can be replaced by Fourier mixing in NLP, and extends this idea to irregular graph-structured data where standard Fourier transforms do not apply.

**The core claim:** You do not need self-attention for competitive graph-level representation learning. A learned spectral operator derived from the graph's own structure, combined with local message passing and per-node gating, is sufficient.

---

## 2. Motivation

FNet (2021) replaced self-attention in Transformers with a fixed 2D FFT over token sequences for NLP. It was fast and competitive. The natural question is: can this work on graphs?

The challenge is that graphs are irregular — they have no fixed grid or sequence structure, so a standard FFT has no natural definition. The solution is to use the graph's own Laplacian eigenvectors as its "Fourier basis." This is mathematically principled: the eigenvectors of the graph Laplacian play the same role as sinusoidal basis functions in classical signal processing.

Prior work (GPS, GraphViT, SAN) uses multi-head self-attention as the global operator. Our work asks: what if we use the graph's spectral structure instead?

---

## 3. Architecture

### 3.1 Overview

```
Input node features
        ↓
SimpleAtomEncoder (embedding lookup per atom feature)
        ↓
+ LapPE (Laplacian positional encoding, k=8 eigenvectors)
        ↓
[Layer 1..4]:
    x_local  = DenseGCNLayer(x, A_norm, edge_attr)
    x_global = SpectralMixMH(x, U, mask)
    gate     = sigmoid(Linear(x))          ← per-node per-feature
    x_mix    = gate * x_local + (1-gate) * x_global
    x        = LayerNorm(x_res + Dropout(x_mix))
        ↓
AttentionPooling (graph-level readout)
        ↓
MLP classifier (Linear → GELU → Dropout → Linear)
```

### 3.2 Key Components

**DenseGCNLayer**
Standard graph convolution with scalar edge feature gating. Edge features are projected to a scalar weight [B,N,N] (NOT [B,N,N,H] — this was an OOM fix) and multiplied element-wise with A_norm before message passing.

```python
E   = sigmoid(edge_lin(edge_attr_dense)).squeeze(-1)  # [B,N,N]
msg = bmm(A_norm * E, x)                              # [B,N,H]
out = LayerNorm(GELU(node_lin(msg)))
```

**SpectralMixMH**
Applies learned spectral filtering via the graph Laplacian eigenbasis U:
1. Project to spectral domain: x_hat = U^T x
2. Apply learned filter: x_filtered = sigmoid(filter_gen(x_hat)) * x_hat
3. Project back: x_out = U * x_filtered

This is the attention-free global mixing operator — the graph's own spectral structure does the mixing, not dot-product attention.

**Per-Node Per-Feature Gating**
Each node independently learns a hidden_dim-dimensional gate vector. This allows node 5 to trust global spectral info 80% while node 12 trusts local GCN info 90%. Strictly more expressive than scalar alpha per layer.

**Sign-Canonicalized Eigendecomposition**
Eigenvectors have sign ambiguity — U and -U are both valid eigenvectors. Without fixing this, LapPE injection is inconsistent across batches and seeds. We canonicalize by ensuring the element with the largest absolute value in each eigenvector column is always positive:

```python
max_abs_idx = abs(U_b).argmax(dim=0)
signs = sign(U_b[max_abs_idx, arange(n)])
signs[signs == 0] = 1.0
U_b = U_b * signs.unsqueeze(0)
```

**Per-Graph Eigendecomposition**
Eigendecomposition is computed on unpadded submatrices (n x n where n = actual node count), then padded back to N x N. This avoids decomposing noise-contaminated padded Laplacians.

### 3.3 Hyperparameters

```
hidden_dim:    128
num_layers:    4
num_heads:     4  (SpectralMixMH)
lap_k:         8  (LapPE eigenvectors)
dropout:       0.1
edge_dim:      3  (bond type, stereo, is_aromatic)
```

### 3.4 Training Setup

```
Optimizer:     AdamW (lr=1e-3, weight_decay=1e-4)
Scheduler:     CosineAnnealingLR (T_max=150, eta_min=1e-5)
Gradient clip: max_norm=1.0
Batch size:    8 (effective 32 via gradient accumulation, accum_steps=4)
Max epochs:    150
Patience:      20 (early stopping on val metric)
Seeds:         [0, 1, 2]
```

---

## 4. Dataset

**Peptides-func** (primary)
- 15,535 molecular graphs of peptides
- Average 150.9 nodes, 307.3 edges per graph
- Task: multi-label binary classification, 10 classes
- Metric: Average Precision (AP), higher is better
- From: Long Range Graph Benchmark (LRGB), Dwivedi et al. 2022
- Specifically designed to test long-range dependency capture

**Peptides-struct** (secondary)
- Same graphs as Peptides-func
- Task: regression of 11 structural properties
- Metric: Mean Absolute Error (MAE), lower is better

Both datasets are from: https://github.com/vijaydwivedi75/lrgb

---

## 5. Results

### 5.1 Main Results — Peptides-func

```
Model                  Params    Test AP ↑
GCN                    508k      0.5930 ±0.0023
GCNII                  505k      0.5543 ±0.0078
GINE                   476k      0.5498 ±0.0079
GatedGCN               509k      0.5864 ±0.0077
GatedGCN+RWSE          506k      0.6069 ±0.0035
Transformer+LapPE      488k      0.6326 ±0.0126
SAN+LapPE              493k      0.6384 ±0.0121
SAN+RWSE               500k      0.6439 ±0.0075   ← SOTA
HybridGraphFNet (ours) 329k      0.6244 ±0.0072
```

Seed breakdown:
- Seed 0: Test AP 0.6270, Best Val 0.6385, Best Epoch 66
- Seed 1: Test AP 0.6317, Best Val 0.6430, Best Epoch 70
- Seed 2: Test AP 0.6146, Best Val 0.6348, Best Epoch 95

### 5.2 Main Results — Peptides-struct

```
Model                  Params    Test MAE ↓
GCN                    508k      0.3496 ±0.0013
GCNII                  505k      0.3471 ±0.0010
GINE                   476k      0.3547 ±0.0045
GatedGCN               509k      0.3420 ±0.0013
GatedGCN+RWSE          506k      0.3357 ±0.0006
Transformer+LapPE      488k      0.2529 ±0.0016   ← SOTA
SAN+LapPE              493k      0.2683 ±0.0043
SAN+RWSE               500k      0.2545 ±0.0012
HybridGraphFNet (ours) 329k      0.2663 ±0.0024
```

Seed breakdown:
- Seed 0: Test MAE 0.2638, Best Val 0.2572, Best Epoch 41
- Seed 1: Test MAE 0.2655, Best Val 0.2614, Best Epoch 43
- Seed 2: Test MAE 0.2696, Best Val 0.2601, Best Epoch 42

### 5.3 Key Observations

1. Our model uses 329k parameters vs 476-509k for all baselines — ~35% fewer params
2. On Peptides-func: we beat all pure GNN baselines and sit competitively with transformer models
3. On Peptides-struct: we beat all pure GNN baselines and match SAN+LapPE (0.2663 vs 0.2683)
4. Same architecture, same hyperparameters for both tasks — only output head changes
5. Low std across seeds (0.0072 func, 0.0024 struct) indicates stable training

---

## 6. Ablation Study

All ablations run on Peptides-func, Seed 0, same training conditions.

### 6.1 Component Ablation

```
Ablation              Test AP    vs Baseline   Interpretation
Full model            0.6270     —
no_lappe              0.6197*    -0.0047       LapPE mildly helpful
no_edge               0.6192     -0.0052       Edge features help moderately
mean_pool             0.6196     -0.0048       Attention pooling marginal gain
scalar_gate           0.5960     -0.0310       Per-node gating critical
no_spectral           0.5941     -0.0329       Spectral mixing critical
```

*no_lappe run with 3 seeds: 0.6197 ±0.0115

### 6.2 Interpretation

The two most critical components are spectral mixing (-0.033) and per-node gating (-0.031). Removing either causes a ~5% absolute drop in AP. They depend on each other: spectral mixing provides global signal, per-node gating controls how each atom routes between local and global. Edge features, LapPE, and attention pooling are secondary, each contributing ~0.005 AP.

The spectral mixing result directly validates the core hypothesis: replacing attention with graph spectral mixing works, and removing it collapses performance to near-GCN levels.

### 6.3 k Sensitivity (LapPE eigenvectors)

```
k     Test AP (Seed 0)
4     0.6289
8     0.6270  ← baseline
16    0.6161
32    0.5989
```

Performance degrades as k increases. Combined with the no_lappe ablation, this suggests the Laplacian eigenbasis used in SpectralMixMH already provides implicit positional information — explicit LapPE injection at high k interferes with this. Small k (4-8) adds marginal benefit; large k is harmful.

---

## 7. Effective Receptive Field (ERF) Analysis

We computed gradient-based ERF for a target node in a test graph (89 nodes, target = node 44).

**Peptides-func:**
- Influence at hop 1: 0.43
- Influence at hop 5: 0.15
- Influence at hop 10: 0.10
- Influence at hop 20: 0.06 (non-zero)

**Peptides-struct:**
- Influence at hop 1: 0.20
- Influence at hop 5: 0.07
- Influence at hop 10: 0.04
- Influence at hop 20: 0.03 (non-zero)

A 4-layer GCN would have exactly zero influence beyond hop 4. Our model maintains non-zero influence at all hops up to the graph diameter, proving the spectral mixing is doing genuine long-range propagation.

The func model shows stronger long-range influence than struct, consistent with the intuition that peptide function depends on global molecular shape while structural properties depend more on local geometry. The model learned task-appropriate receptive fields from data alone — no architectural changes between tasks.

---

## 8. Efficiency

```
Parameters:     329k (vs 476-509k baselines, ~35% fewer)
Peak memory:    ~222 MB per seed
Time per epoch: ~72s on Kaggle T4 GPU (batch_size=8, accum_steps=4)
Time per seed:  ~6,000s average (~1.7 hrs)
```

No attention computation → O(N²) attention matrix never materializes. The dominant cost is the per-graph eigendecomposition O(N³), which runs on CPU-equivalent small matrices (~150 nodes average).

---

## 9. Critical: The LRGB Reassessment Paper

**MUST READ before writing the paper.**

Tönshoff et al. (2023), "Where Did the Gap Go? Reassessing the Long-Range Graph Benchmark"
- arXiv: https://arxiv.org/pdf/2309.00367
- OpenReview: https://openreview.net/pdf?id=rIUjwxc5lj

### What They Found

With proper hyperparameter tuning (deeper prediction head, weight_decay=0.0, feature normalization, skip connections), GCN reaches **0.6860 AP on Peptides-func** — beating our 0.6244 and all graph transformer baselines. The improvement on Peptides-struct is even more dramatic, with a tuned GCN closing the gap entirely to transformer models.

The key finding: most LRGB baseline results were suboptimal due to a single linear prediction head. Switching to a multi-layer head with the same hidden dimension drove most of the improvement.

### Does This Hurt Us?

**Not significantly, if handled correctly.** Here is why:

1. Our model already has a 2-layer MLP prediction head — we already benefited from this fix without knowing about the paper. Our 0.6244 is not an undertuned result in the same way the original baselines were.

2. The reassessment paper applied tuning to ALL models including transformers — those results also moved. The relative landscape shifts for everyone, not just GNNs.

3. Our contribution is the **attention-free spectral mixing mechanism** — the architecture novelty — not just a number on a leaderboard. Even if a tuned GCN beats us numerically, it doesn't invalidate a novel architectural approach.

4. We were not subject to exhaustive hyperparameter search. Our 0.6244 is a single configuration across 3 seeds. Tuning would likely improve our numbers too.

### How to Handle This in the Paper

Cite the reassessment paper honestly and use this framing:

> "We compare against originally reported LRGB baselines following standard evaluation protocol (Dwivedi et al., 2022). Tönshoff et al. (2023) demonstrate that LRGB baselines are sensitive to hyperparameter choices, with tuned MPGNNs closing the gap to graph transformers. Our model achieves competitive performance under the standard evaluation protocol without exhaustive tuning, and our architecture uses a 2-layer MLP prediction head consistent with their recommended practice. Further gains through systematic hyperparameter optimization are left to future work."

### What NOT to Do

Do not ignore this paper — reviewers at LoG and any graph learning venue will know about it and will ask why you didn't cite it. Acknowledging it proactively is far better than having a reviewer raise it.

Do not claim to beat GNN baselines in the abstract without noting the reassessment context.

---

## 10. What We Did NOT Find (Honest Limitations)

1. **LapPE is mildly redundant**: The spectral mixing already encodes positional structure implicitly. LapPE helps by only ~0.005 AP.
2. **k=4 slightly beats k=8 in one seed**: Not conclusive — within seed variance. We report k=8 as baseline since it has 3-seed results.
3. **We don't beat SOTA**: SAN+RWSE (0.6439) and Transformer+LapPE (0.2529) are better on their respective tasks under standard evaluation. Tuned GCN (Tönshoff et al. 2023) reaches 0.6860, above our 0.6244.
4. **Scalability**: Per-graph eigendecomposition is O(N³) — not suitable for large graphs (e.g., ogbn-arxiv). We only evaluated on medium-scale molecular graphs.
5. **No hyperparameter search**: Our results use a single configuration. Systematic tuning would likely improve performance further.

---

## 10. Suggested Paper Structure

### Title
HybridGraphFNet: Attention-Free Graph Learning via Learned Spectral-Local Gating

### Abstract (key points to hit)
- Motivation: FNet showed attention is replaceable in NLP; we extend to graphs
- Method: Laplacian eigenbasis as mixing operator + GCN + per-node gating
- Results: competitive with transformer-based models, 35% fewer parameters
- Finding: spectral mixing and per-node gating are jointly critical (ablation)

### Sections
1. Introduction — FNet motivation, graph learning challenge, our contribution
2. Related Work — FNet, GPS, GraphViT, SAN, spectral GNNs, LapPE
3. Method — full architecture description (use Section 3 above)
4. Experiments
   - 4.1 Datasets and setup
   - 4.2 Main results (Table 1: func + struct vs baselines)
   - 4.3 Ablation study (Table 2: component ablation)
   - 4.4 k sensitivity (Table 3 or Figure)
   - 4.5 ERF analysis (Figure: bar charts)
5. Discussion — parameter efficiency, task-adaptive ERF, limitations
6. Conclusion

### Key Citations Needed
- FNet: Lee-Thorp et al. 2021
- LRGB benchmark: Dwivedi et al. 2022
- LRGB reassessment: Tönshoff et al. 2023 (MUST CITE — arxiv 2309.00367)
- GPS: Rampasek et al. 2022
- SAN: Kreuzer et al. 2021
- Laplacian PE: Dwivedi & Bresson 2021
- GCN: Kipf & Welling 2017
- GINE: Hu et al. 2020
- GatedGCN: Bresson & Laurent 2017

---

## 11. Things Still Pending

- [ ] no_local ablation (3 seeds) — spectral-only baseline
- [ ] no_spectral ablation (seeds 1,2) — GCN-only baseline with 3 seeds
- [ ] These two form the "mixing strategy" table showing progressive improvement

Once those finish, the complete results table will be:

```
GCN only (no_spectral)     X.XXX ±X.XXX
Spectral only (no_local)   X.XXX ±X.XXX
Hybrid + scalar gate        0.5960 (seed 0 only)
Hybrid + per-node gate      0.6244 ±0.0072   ← final
```

---

## 12. Code Structure

All code written in PyTorch + PyTorch Geometric.

Key classes:
- `SimpleAtomEncoder` — embedding lookup for 9 atom features
- `DenseGCNLayer` — GCN with scalar edge gating
- `SpectralMixMH` — spectral mixing via Laplacian eigenbasis
- `AttentionPooling` — graph readout
- `HybridGraphFNet_Best` — full model with ablation flag support
- `HybridGraphFNet_Best_Peptides` — wrapper with atom encoder

Ablation flag options: `None, "no_lappe", "no_edge", "scalar_gate", "mean_pool", "no_spectral", "no_local"`

Training function: `train_model()` with cosine LR, grad clip, grad accumulation, gate health check, checkpoint saving, single clean test evaluation at end of each seed.

---

## 13. Reproducibility Checklist

- 3 seeds for all main results
- Best checkpoint saved per seed, test evaluated once after training
- Val used for model selection only — test never seen during training
- Gradient accumulation ensures effective batch = 32 despite batch_size = 8
- Sign canonicalization of eigenvectors ensures cross-batch/seed consistency
- Per-graph eigendecomp on unpadded submatrices — no padded noise
