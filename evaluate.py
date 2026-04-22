import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_recall_curve, roc_curve, auc, average_precision_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize

# Import utility functions from training script
from train import load_manifest, iter_graph_batches_from_paths
from model import GraphClassifier

def load_trained_model(
    model_path,
    device,
    node_dim=1280,
    hidden_dim=256,
    num_heads=4,
    dropout=0.5,
    num_classes=5,
    use_gat=True,
):
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # 1) If the whole module was saved
    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint.to(device)
        model.eval()
        return model

    # 2) If it's a state dict or a dictionary containing the state dict
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model = GraphClassifier(
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        num_attention_heads=num_heads,
        dropout=dropout,
        num_classes=num_classes,
        use_gat=use_gat,
    ).to(device)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[Warning] missing keys: {missing}")
        print(f"[Warning] unexpected keys: {unexpected}")

    model.eval()
    return model


def evaluate_model(model, manifest_path, batch_size, device, mc_dropout=True, mc_passes=30, entropy_thresh=1.5):
    print(f"Loading data from manifest: {manifest_path}")
    _, batch_paths = load_manifest(manifest_path)
    
    all_labels = []
    all_preds = []
    all_probs = []
    total_items = 0

    if mc_dropout:
        model.train() # Enable dropout during evaluation
    else:
        model.eval()

    with torch.no_grad():
        for batch_file, graphs in iter_graph_batches_from_paths(batch_paths):
            if not graphs:
                continue

            loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
            for pyg_batch in loader:
                pyg_batch = pyg_batch.to(device)
                y = pyg_batch.y.view(-1).long()
                
                if mc_dropout:
                    # Run N scholastic forward passes
                    batch_probs = []
                    for _ in range(mc_passes):
                        logits = model(pyg_batch)
                        batch_probs.append(torch.softmax(logits, dim=1))
                    batch_probs = torch.stack(batch_probs) # [mc_passes, B, num_classes]
                    mean_probs = batch_probs.mean(dim=0)
                    
                    # Compute entropy as uncertainty
                    entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=1)
                    
                    preds = mean_probs.argmax(dim=1)
                    # Reject low-confidence inputs
                    preds[entropy > entropy_thresh] = -1
                    probs = mean_probs
                else:
                    logits = model(pyg_batch)
                    probs = torch.softmax(logits, dim=1)
                    preds = probs.argmax(dim=1)

                bs = int(y.shape[0])
                total_items += bs
                all_labels.extend(y.cpu().numpy().tolist())
                all_preds.extend(preds.cpu().numpy().tolist())
                all_probs.extend(probs.cpu().numpy().tolist())

            del loader
            del graphs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    model.eval()
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_conf_matrix(labels, preds, classes, output_path='confusion_matrix.png'):
    cm = confusion_matrix(labels, preds, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(c) for c in classes])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig(output_path)
    plt.close()
    print(f"Saved Confusion Matrix to {output_path}")


# Deprecating plot_precision_recall, plot_roc for strict multi-class currently due to dimension mismatch. Macro aggregation computed numerically instead.


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained GNN model on a test set.")
    parser.add_argument('--model-path', default='gnn_model.pt', help='Path to the trained model')
    parser.add_argument('--manifest-path', default='test_graph_batches/manifest.json', help='Path to the test graph batches manifest.json')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for evaluation')
    parser.add_argument('--device', default=None, help='Device to use (e.g. cpu, cuda)')
    parser.add_argument('--mc-dropout', action='store_true', help='Enable MC Dropout for uncertainty evaluation')
    parser.add_argument('--mc-passes', type=int, default=30, help='Number of MC Dropout passes')
    parser.add_argument('--entropy-thresh', type=float, default=1.5, help='Entropy threshold to assign -1 / Unknown class label')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Model hidden dimension')
    parser.add_argument('--num-heads', type=int, default=4, help='Model attention heads')
    
    args = parser.parse_args()

    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        return
        
    if not os.path.exists(args.manifest_path):
        print(f"Error: Test data manifest not found at {args.manifest_path}")
        print("You might need to generate the test graph batches first using:")
        print(f"  python binding_site_graph.py --split-file test.txt --output-dir {os.path.dirname(args.manifest_path)}")
        return

    # Load model
    model = load_trained_model(
        model_path=args.model_path,
        device=device,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads
    )

    # Evaluate
    labels, preds, probs = evaluate_model(
        model=model,
        manifest_path=args.manifest_path,
        batch_size=args.batch_size,
        device=device,
        mc_dropout=args.mc_dropout,
        mc_passes=args.mc_passes,
        entropy_thresh=args.entropy_thresh,
    )

    if len(labels) == 0:
        print("No evaluation data found.")
        return

    # Map unknown labels into valid evaluation format
    valid_mask = labels != -1
    if len(labels[valid_mask]) == 0:
        print("No valid evaluation labels left after applying mask.")
        return

    labels_clean = labels[valid_mask]
    preds_clean = preds[valid_mask]
    
    # Identify unique valid classes seen in this set (including the -1 reject category if it occurred)
    classes = sorted(list(set(labels_clean).union(set(preds_clean))))
    
    # Compute multi-class metrics (excluding the rejection class from F1 target average potentially, but let's do macro)
    f1 = f1_score(labels_clean, preds_clean, average='macro', zero_division=0)
    
    # Check how many were rejected
    num_rejected = np.sum(preds == -1)
    
    print("\n" + "="*30)
    print("EVALUATION RESULTS")
    print("="*30)
    print(f"Total test samples : {len(labels)}")
    print(f"Valid label samples: {len(labels_clean)}")
    print(f"Rejected inputs (-1): {num_rejected}")
    print(f"Macro F1 Score     : {f1:.4f}")
    print("="*30 + "\n")

    # Generate Confusion Matrix
    plot_conf_matrix(labels_clean, preds_clean, classes, output_path='confusion_matrix_test.png')

    with open("evaluation_results_summary.txt", "w") as f:
        f.write("EVALUATION RESULTS\n")
        f.write("="*30 + "\n")
        f.write(f"Total test samples : {len(labels)}\n")
        f.write(f"Valid label samples: {len(labels_clean)}\n")
        f.write(f"Rejected inputs (-1): {num_rejected}\n")
        f.write(f"Macro F1 Score     : {f1:.4f}\n")
        f.write("="*30 + "\n")

if __name__ == '__main__':
    main()
