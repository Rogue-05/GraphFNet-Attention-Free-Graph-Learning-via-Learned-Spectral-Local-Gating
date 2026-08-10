# GraphFNet: Attention-Free Graph Learning via Learned Spectral-Local Gating

This repository contains the official, anonymized implementation for the paper **"GraphFNet: Attention-Free Graph Learning via Learned Spectral-Local Gating"** submitted to CIKM 2026.

GraphFNet is a parameter-efficient, memory-efficient alternative to Graph Transformers. By replacing the dynamic $\mathcal{O}(N^2)$ memory footprint of self-attention with a global spectral mixing operator derived from a graph Laplacian eigenbasis, the architecture dramatically reduces parameters while maintaining long-range signal propagation capability.

---

## 📂 Repository Structure

* `graphfnet.ipynb`: Main notebook containing the core architecture backbone, data loading pipeline, and training routines for the LRGB Peptides-func and Peptides-struct tasks (Main Results, Section 4.3).
* `graphfnet_ablations.ipynb` & `ablations_results.csv`: Reproducible code and logged outputs for the systematic module ablation study and $k$-sensitivity analysis (Section 4.4).
* `erf_analysis.py`: Script to compute gradient-based node influence, generating `erf_func.png` and `erf_struct.png` for Effective Receptive Field diagnostic probing (Section 4.5).
* `graphFNet_neighbors_match.ipynb` & `neighbors_match_results.csv` / `.png`: Synthetic evaluation, tree generation, and mechanistic gate/spectral analysis on the Tree-NeighborsMatch task up to radius $r=5$ (Section 4.6).
* `graphfnet_bbbp.py`: Auxiliary evaluation on the MoleculeNet BBBP dataset demonstrating the strict Bemis-Murcko scaffold split and the **offline static spectral caching strategy** (Section 4.8).

---

## 🛠️ Environment & Installation

The implementation relies on PyTorch, PyTorch Geometric (PyG), and RDKit. To replicate our environment, configure a Python 3.10+ environment with the following dependencies:

```bash
pip install torch torch-geometric scikit-learn rdkit tqdm numpy pandas matplotlib


Save the file.

---

### 4. Check what you changed

In the terminal:

```bash
git diff
