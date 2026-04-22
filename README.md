# Structure-Based Cofactor Predictor

This repository contains an end-to-end pipeline for predicting cofactor-protein interactions (e.g., binding sites for NAD, FAD, ATP) using structural data. 

It leverages **PyTorch Geometric (PyG)** to train Graph Neural Networks (GNNs) on protein-ligand interaction graphs, with **ESM-2** providing deep sequence representations for protein residues.

## Pipeline Overview

The pipeline consists of the following modular steps:

1. **Binding Site Extractions (`Binding_site_ex.py`)**  
   Iterates through the original structure files to isolate protein-ligand binding pockets under a specific threshold distance (e.g., 6.0 Å). It natively parses the specific structural cofactor molecule (`actual_ligand_name`) and converts it directly into a **multi-class integer label** mapping across known types (`0: NAD, 1: FAD, 2: ATP, 3: ADP, 4: COA`).
   Computes:
   - Protein-Ligand contacts
   - In-ligand covalent bonds
   - Extract contact maps for surrounding residues
   *Outputs a consolidated `binding_sites_by_protein.json`.*

2. **Structure Clustering (`structure_clustering.py`)**  
   Splits dataset `.pdb` files into distinct `train.txt`, `validation.txt`, and `test.txt` partitions. It leverages [DataSAIL](https://github.com/kalininalab/DataSAIL) connected with Foldseek (`tmscore`) to ensure robust independent structure-based 3D clusters. 
   **Note**: It intelligently reads the extracted multiclass labels natively through the JSON to stratify clusters evenly across all 5 distinct cofactor classes!

3. **Graph Generation (`binding_site_graph.py`)**  
   Transforms the extracted dictionaries into PyTorch Geometric `Data` graphs. 
   - Dynamically calls `esm2_feature_ex.py` to embed protein sequences using HuggingFace's ESM-2.
   - Embeds ligand representations and defines heterogenous node/edge types.
   - Multiclass labels are correctly loaded tracking the distinct cofactors.
   *Outputs serialized PyTorch datasets grouped into `manifest.json`.*

4. **Model Architecture (`model.py`)**  
   A flexible graph, 5-class classifier.
   - Separate embedding projections for Protein (ESM) and Ligand nodes.
   - Message Passing Neural Network using flexible Multi-Head **GATv2**.
   - Features Pre-Norm configurations and residual connections.
   - Includes custom structural attention pooling over protein nodes exclusively.

5. **Training (`train.py`)**  
   Takes the generated `manifest.json` from the serialized batches. Includes:
   - Early stopping hooks.
   - Streaming memory-safe batch loading.
   - AdamW optimizer logging multi-class **Macro F1 Score** and Cross-Entropy loss metrics.

6. **Evaluation (`evaluate.py`)**  
   Loads the model checkpoint (`gnn_model.pt`) and performs validation set sweeps, producing robust scikit-learn metrics. Autogenerates:
   - Confusion Matrices (`confusion_matrix_test.png`)
   
   **Uncertainty Scoring & Rejection**: The evaluation script supports Monte Carlo (MC) Dropout passes. By adding `--mc-dropout`, the model will predict your input stochastically across independent runs (`--mc-passes 30`). By evaluating the cross-prediction entropy, highly-erratic structural predictors are cleanly rejected to the `Unknown / Non-Binder` `-1` class!

- `PDB/` - Should contain target structure files locally uncompressed.
- `binding_site_graph_batches/` - Auto-generated cached serialized batches built prior to training.
- Configuration lists (`train.txt`, etc.) dictate specific subsets to run at any particular lifecycle step.

## Typical Usage
1. Provide the source structures inside flat folders in `PDB/`.
2. Identify binding pockets & map known labels natively across multi-class bindings: `python Binding_site_ex.py` 
3. Generate stratified structural dataset splits across the classes: `python structure_clustering.py`
4. Cast to Graphs in batches: `python binding_site_graph.py`
5. Run the model: `python train.py --epochs 25`
6. Verify final test metrics robustly handling outliers via Monte-Carlo uncertainty limits: `python evaluate.py --mc-dropout`
