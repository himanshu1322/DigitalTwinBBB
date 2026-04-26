import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from models.moe_layer import BBBMoELayer
from utils.data_loader import get_stratified_loaders

def evaluate_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dim, num_experts = 2055, 16
    
    # 1. Load Data & Model
    _, test_loader = get_stratified_loaders('data/B3DB_classification.tsv', batch_size=64, dim=dim)
    model = BBBMoELayer(dim=dim, num_experts=num_experts, k=2).to(device)
    model.load_state_dict(torch.load("models/bbb_ternary_moe_95.pth"))
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for vectors, labels in test_loader:
            outputs = model(vectors.to(device)).mean(dim=1)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    # 2. Generate Metrics
    print("\n📊 --- FINAL RESEARCH METRICS ---")
    print(classification_report(all_labels, all_preds, target_names=['Blocked (BBB-)', 'Passed (BBB+)']))
    
    auc = roc_auc_score(all_labels, all_probs)
    print(f"📈 AUC-ROC Score: {auc:.4f}")
    
    # 3. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\n🧩 Confusion Matrix:")
    print(f"True Negatives (Correctly Blocked): {cm[0][0]}")
    print(f"False Positives (Mistakenly Passed): {cm[0][1]} <-- This is where Dopamine lives")
    print(f"False Negatives (Mistakenly Blocked): {cm[1][0]}")
    print(f"True Positives (Correctly Passed): {cm[1][1]}")

if __name__ == "__main__":
    evaluate_model()