# Structure-Based Cofactor Predictor

This repository contains an end-to-end pipeline for predicting cofactor-protein interactions (e.g., binding sites for NAD, FAD, ATP) using structural data. 

It leverages **PyTorch Geometric (PyG)** to train Graph Neural Networks (GNNs) on protein-ligand interaction graphs, with **ESM-2** providing deep sequence representations for protein residues.

## Pipeline Overview

The pipeline consists of the following modular steps:

1. **Structure Clustering (`structure_clustering.py`)**  
   Splits dataset `.pdb` files into distinct `train.txt`, `validation.txt`, and `test.txt` partitions. It uses [DataSAIL](https://github.com/kalininalab/DataSAIL) connected with Foldseek (`tmscore`) to ensure robust independent structure-based 3D clusters (reducing data leakage among structurally identical folds).

2. **Binding Site Extractions (`Binding_site_ex.py`)**  
   Iterates through PDB files (organized in `positive_samole` and `negative_sample` folders) to isolate protein-ligand binding pockets under a specific threshold distance (e.g., 6.0 Å). Computes:
   - Protein-Ligand contacts
   - In-ligand covalent bonds
   - Extract contact maps for surrounding residues
   *Outputs a consolidated `binding_sites_by_protein.json`.*

3. **Graph Generation (`binding_site_graph.py`)**  
   Transforms the extracted dictionaries into PyTorch Geometric `Data` graphs. 
   - Dynamically calls `esm2_feature_ex.py` to embed protein sequences using HuggingFace's ESM-2.
   - Embeds ligand representations and defines heterogenous node/edge types.
   - Labels are automatically assigned by tracing paths from the PDB pipeline.
   *Outputs serialized PyTorch datasets grouped into `manifest.json`.*

4. **Model Architecture (`model.py`)**  
   A flexible graph classifier.
   - Separate embedding projections for Protein (ESM) and Ligand nodes.
   - Message Passing Neural Network using flexible Multi-Head **GATv2**.
   - Features Pre-Norm configurations and residual connections.
   - Includes custom structural attention pooling over protein nodes exclusively.

5. **Training (`train.py`)**  
   Takes the generated `manifest.json` from the serialized batches. Includes:
   - Early stopping hooks.
   - Streaming memory-safe batch loading.
   - AdamW optimizer logging Average Precision (AP) and binary cross-entropy loss metrics.

6. **Evaluation (`evaluate.py`)**  
   Loads the model checkpoint (`gnn_model.pt`) and performs validation set sweeps, producing robust scikit-learn metrics. Autogenerates:
   - Precision-Recall curves (`precision_recall_curve_test.png`)
   - ROC curves (`roc_curve_test.png`)

## Directory Layout
- `PDB/` - Should contain target structure files split by parent classification (e.g., `positive_samole/` and `negative_sample/`).
- `binding_site_graph_batches/` - Auto-generated cached serialized batches built prior to training.
- Configuration lists (`train.txt`, etc.) dictate specific subsets to run at any particular lifecycle step.

## Typical Usage
1. Provide the source structures in `PDB/`.
2. Generate splits: `python structure_clustering.py`
3. Identify binding pockets: `python Binding_site_ex.py --ligand-name NAD` 
4. Cast to Graphs in batches: `python binding_site_graph.py`
5. Run the model: `python train.py --epochs 25`
6. Verify final test metrics: `python evaluate.py`
