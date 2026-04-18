import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_recall_curve, roc_curve, auc, average_precision_score, confusion_matrix
from torch_geometric.loader import DataLoader

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
    num_classes=2,
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


def evaluate_model(model, manifest_path, batch_size, device):
    print(f"Loading data from manifest: {manifest_path}")
    _, batch_paths = load_manifest(manifest_path)
    
    all_labels = []
    all_preds = []
    total_items = 0

    with torch.no_grad():
        for batch_file, graphs in iter_graph_batches_from_paths(batch_paths):
            if not graphs:
                continue

            loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
            for pyg_batch in loader:
                pyg_batch = pyg_batch.to(device)
                y = pyg_batch.y.view(-1).long()
                logits = model(pyg_batch)

                # Get probability for class 1 (positive interaction)
                probs = torch.softmax(logits, dim=1)[:, 1]

                bs = int(y.shape[0])
                total_items += bs
                all_labels.extend(y.cpu().numpy().tolist())
                all_preds.extend(probs.cpu().numpy().tolist())

            del loader
            del graphs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return np.array(all_labels), np.array(all_preds)


def plot_precision_recall(labels, preds, output_path='precision_recall_curve.png'):
    precision, recall, _ = precision_recall_curve(labels, preds)
    ap_score = average_precision_score(labels, preds)
    
    plt.figure()
    plt.plot(recall, precision, marker='.', label=f'AP = {ap_score:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved Precision-Recall curve to {output_path}")


def plot_roc(labels, preds, output_path='roc_curve.png'):
    fpr, tpr, _ = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, marker='.', label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved ROC curve to {output_path}")

def calc_conf_metrix(labels, preds, threshold=0.5):
    predicted_classes = (preds > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted_classes).ravel()
    return tn, fp, fn, tp


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained GNN model on a test set.")
    parser.add_argument('--model-path', default='gnn_model.pt', help='Path to the trained model')
    parser.add_argument('--manifest-path', default='test_graph_batches/manifest.json', help='Path to the test graph batches manifest.json')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for evaluation')
    parser.add_argument('--device', default=None, help='Device to use (e.g. cpu, cuda)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Probability threshold for F1 score computing')
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
    labels, preds = evaluate_model(
        model=model,
        manifest_path=args.manifest_path,
        batch_size=args.batch_size,
        device=device
    )

    if len(labels) == 0:
        print("No evaluation data found.")
        return

    # Compute metrics
    ap_score = average_precision_score(labels, preds)
    predicted_classes = (preds > args.threshold).astype(int)
    f1 = f1_score(labels, predicted_classes)
    tn, fp, fn, tp = calc_conf_metrix(labels, preds, args.threshold)
    
    print("\n" + "="*30)
    print("EVALUATION RESULTS")
    print("="*30)
    print(f"Total test samples : {len(labels)}")
    print(f"Average Precision (AP): {ap_score:.4f}")
    print(f"F1 Score (thresh={args.threshold:.2f}): {f1:.4f}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Positives (TP): {tp}")
    print("="*30 + "\n")

    # Generate plots
    plot_precision_recall(labels, preds, output_path='precision_recall_curve_test.png')
    plot_roc(labels, preds, output_path='roc_curve_test.png')


    with open("evaluation_results_summary.txt", "w") as f:
        f.write("EVALUATION RESULTS\n")
        f.write("="*30 + "\n")
        f.write(f"Total test samples : {len(labels)}\n")
        f.write(f"Average Precision (AP): {ap_score:.4f}\n")
        f.write(f"F1 Score (thresh={args.threshold:.2f}): {f1:.4f}\n")
        f.write(f"True Negatives (TN): {tn}\n")
        f.write(f"False Positives (FP): {fp}\n")
        f.write(f"False Negatives (FN): {fn}\n")
        f.write(f"True Positives (TP): {tp}\n")
        f.write("="*30 + "\n")

if __name__ == '__main__':
    main()
