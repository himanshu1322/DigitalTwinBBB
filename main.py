import torch
import torch.nn as nn
import torch.optim as optim
from models.moe_layer import BBBMoELayer
from utils.data_loader import get_stratified_loaders

def train_to_95():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Note the new dimension: 2048 + 3
    dim, num_experts, epochs = 2055, 16, 50 
    
    train_loader, test_loader = get_stratified_loaders('data/B3DB_classification.tsv', batch_size=64, dim=dim)
    model = BBBMoELayer(dim=dim, num_experts=num_experts).to(device)
    
    # Label Smoothing: Helps 1.58-bit models generalize better
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.2]).to(device)) 
    optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.05)
    
    # Cosine Annealing with Warmup
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, 
                                            steps_per_epoch=len(train_loader), 
                                            epochs=epochs)

    print(f"🧬 Starting Final Research Run...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for vectors, labels in train_loader:
            vectors, labels = vectors.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(vectors).mean(dim=1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            # Gradient Clipping: Prevents 1.58-bit spikes
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            # Quick Validation
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for v, l in test_loader:
                    out = model(v.to(device)).mean(dim=1)
                    pred = (torch.sigmoid(out) > 0.5).float()
                    correct += (pred == l.to(device)).sum().item()
                    total += l.size(0)
            print(f"Epoch {epoch+1} | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {correct/total:.2%}")

    torch.save(model.state_dict(), "models/bbb_ternary_moe_95.pth")
    print("🏆 Training Complete. High-Accuracy Weights Saved.")

if __name__ == "__main__":
    train_to_95()