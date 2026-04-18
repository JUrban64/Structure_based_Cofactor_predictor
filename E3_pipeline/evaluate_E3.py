import sys
import os
import argparse
import torch

# Nastavit cesty pro originální pipeline
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import evaluate
from model_E3 import GraphClassifierE3

# Mimořádně dynamicky nahradíme používaný model originálním scriptem 
# za náš nový invariantní E(3)
evaluate.GraphClassifier = GraphClassifierE3

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate Trained E(3)-Equivariant GNN'
    )
    # Default nastavení na E3 verzi generovaných předtím v train_E3.py
    parser.add_argument(
        '--manifest-path',
        default='../binding_site_graph_test/manifest.json',
    )
    parser.add_argument(
        '--model-path',
        default='gnn_model_best_e3.pt',
    )
    parser.add_argument('--batch-size', type=int, default=4)
    # Pamětová úspora pro E(3) model (byla např. 64 defaultově)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--device', default=None)
    parser.add_argument('--threshold', type=float, default=0.5)
    
    args = parser.parse_args()

    device = args.device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    if not os.path.exists(args.manifest_path):
        print(f"File not found: {args.manifest_path}")
        sys.exit(1)

    # Load E(3) model
    model = evaluate.load_trained_model(
        model_path=args.model_path,
        device=device,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads
    )

    # Evaluate pomocí originální funkce
    labels, preds = evaluate.evaluate_model(
        model=model,
        manifest_path=args.manifest_path,
        batch_size=args.batch_size,
        device=device
    )

    if len(labels) == 0:
        print("No evaluation data found.")
        sys.exit(0)

    # Compute metrics přes originální evaluate pipeline
    ap_score = evaluate.average_precision_score(labels, preds)
    predicted_classes = (preds > args.threshold).astype(int)
    f1 = evaluate.f1_score(labels, predicted_classes)
    tn, fp, fn, tp = evaluate.calc_conf_metrix(labels, preds, args.threshold)
    
    print("\n" + "="*30)
    print("EVALUATION RESULTS E(3) EQUIVARIANT GNN")
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
    evaluate.plot_precision_recall(labels, preds, output_path='precision_recall_curve_e3_test.png')
    evaluate.plot_roc(labels, preds, output_path='roc_curve_e3_test.png')

    with open("evaluation_results_summary_e3.txt", "w") as f:
        f.write("EVALUATION RESULTS E3 EQUIVARIANT GNN\n")
        f.write("="*30 + "\n")
        f.write(f"Total test samples : {len(labels)}\n")
        f.write(f"Average Precision (AP): {ap_score:.4f}\n")
        f.write(f"F1 Score (thresh={args.threshold:.2f}): {f1:.4f}\n")
        f.write(f"True Negatives (TN): {tn}\n")
        f.write(f"False Positives (FP): {fp}\n")
        f.write(f"False Negatives (FN): {fn}\n")
        f.write(f"True Positives (TP): {tp}\n")
        f.write("="*30 + "\n")
