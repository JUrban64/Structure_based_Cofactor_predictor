from sklearn.metrics import f1_score, precision_recall_curve, roc_curve
from matplotlib import pyplot as plt
import numpy as np
import torch
from train import GraphClassifier

def load_model(
    model_path,
    device="cpu",
    node_dim=1280,
    hidden_dim=256,
    num_attention_heads=4,
    dropout=0.5,
    num_classes=2,
    use_gat=True,
):
    device = torch.device(device)
    checkpoint = torch.load(model_path, map_location=device)

    # 1) Ulozen cely modul
    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint.to(device)
        model.eval()
        return model

    # 2) Ulozen state_dict
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict", checkpoint)
        model = GraphClassifier(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            num_classes=num_classes,
            use_gat=use_gat,
        ).to(device)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[load_model] missing keys: {missing}")
            print(f"[load_model] unexpected keys: {unexpected}")
        model.eval()
        return model

    raise TypeError(f"Nepodporovany typ checkpointu: {type(checkpoint)}")



def evaluate_model(model, dataloader):
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            outputs = model(inputs)
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
    
    return np.array(all_labels), np.array(all_preds) 


def plot_precision_recall(labels, preds):
    precision, recall, _ = precision_recall_curve(labels, preds)
    plt.figure()
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig('precision_recall_curve.png')
    plt.close()

def plot_roc(labels, preds):
    fpr, tpr, _ = roc_curve(labels, preds)
    plt.figure()
    plt.plot(fpr, tpr, marker='.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.savefig('roc_curve.png')
    plt.savefig('roc_curve.pdf')
    plt.close()



def main():
    model_path = 'best_model.pth'
    model = load_model(model_path)
    
    # Předpokládáme, že dataloader je definován a připraven
    dataloader = ...  # TODO: vytvořit dataloader pro testovací data
    
    labels, preds = evaluate_model(model, dataloader)
    
    plot_precision_recall(labels, preds)
    plot_roc(labels, preds)
    
    print(f"F1 Score: {f1_score(labels, preds > 0.5)}")

if __name__ == "__main__":
    main()


