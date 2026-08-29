# GraphFNet: Attention-Free Graph Learning via Learned Spectral-Local Gating

Official research repository for **GraphFNet**, an attention-free graph neural network architecture that replaces the quadratic memory footprint of dense self-attention with learned global spectral mixing over the graph Laplacian eigenbasis, adaptively gated with local message passing.

---

## 📁 Repository Structure

```
├── paper/                                  # LaTeX source & figures for TMLR submission
│   ├── main.tex                            # Main paper LaTeX source
│   ├── sample-base.bib                     # Bibliography file
│   ├── math_commands.tex                   # Mathematical notation macros
│   ├── tmlr.sty / tmlr.bst                 # TMLR template style files
│   └── *.png                               # Figures (ERF, VRAM scaling, gate analysis, etc.)
│
├── new/                                    # Core models, experiment notebooks & scripts
│   ├── model.py                            # PyTorch model definitions (GraphFNet backbone)
│   ├── graphfnet.ipynb                     # Peptides-func main training notebook
│   ├── graphFnet_peptide_struct.ipynb      # Peptides-struct regression notebook
│   ├── graphfnet-hcp-gender.ipynb          # NeuroGraph HCP-Gender classification
│   ├── graphfnet-hcp-task.ipynb            # NeuroGraph HCP-Task cognitive state classification
│   ├── graphfnet_pascalvoc_sp.ipynb        # PascalVOC-SP superpixel node segmentation
│   ├── graphfnet_ablations.ipynb           # Systematic architecture ablation studies
│   ├── graphFnet_neighbors_match.ipynb     # Tree-NeighborsMatch mechanistic diagnostic
│   ├── GraphFnet_Gate_Analysis.ipynb       # Emergent spectral-to-local gate analysis
│   ├── erf_analysis.py                     # Effective Receptive Field (ERF) diagnostic
│   └── vram_scaling_benchmark_extended.ipynb # 5-model VRAM & throughput scaling suite
│
├── paper_notes.md                          # Technical notes, raw metrics & experimental logs
├── .gitignore                              # Git ignore rules for LaTeX builds, caches & binaries
└── README.md                               # Repository overview & documentation
```

---

## 🚀 Key Highlights & Results

- **Attention-Free Global Mixing**: Replaces dynamic $\mathcal{O}(N^2)$ pairwise self-attention with static linear spectral projections ($\mathcal{O}(N \cdot H)$ activation memory), reducing forward-pass memory by up to **$102.7\times$** at $N=10{,}000$ and **$14.0\times$** in full 4-layer models at $N=5{,}000$.
- **NeuroGraph Connectomics ($N=1{,}000$)**:
  - **HCP-Gender**: **81.17% ± 1.90%** test accuracy (State-of-the-Art; outperforming GraphGPS at 76.85%, Graph-Mamba at 77.16%, and BrainMAP at 78.92%).
  - **HCP-Task**: **93.74% ± 0.54%** test accuracy on 7-class whole-brain cognitive task classification.
- **Long Range Graph Benchmark (LRGB Peptides)**:
  - Matches standard dense-attention Graph Transformers (0.6244 AP on Peptides-func, 0.2663 MAE on Peptides-struct) while requiring **$\sim$35% fewer parameters** (329k vs. 500k budget) and under 250 MB peak memory.
- **Emergent Interpretability**: Autonomous spectral-to-local routing curriculum across layers (early layers capture macroscopic topology; later layers refine local chemistry).

---

## 🛠️ Environment Setup

```bash
# Clone the repository
git clone <repo-url>
cd graph

# Create and activate environment
conda create -n graphfnet python=3.10 -y
conda activate graphfnet

# Install PyTorch & PyG dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric
pip install numpy networkx scikit-learn matplotlib tqdm
```

---

