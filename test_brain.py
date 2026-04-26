import torch
from models.moe_layer import BBBMoELayer
from utils.featurizer import smiles_to_vector

def run_test(smiles_string, name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dim = 2055 
    
    # Update num_experts to 16 and k to 2 to match the trained model
    model = BBBMoELayer(dim=dim, num_experts=16, k=2).to(device)
    
    # Load the weights
    model.load_state_dict(torch.load("models/bbb_ternary_moe_95.pth"))
    model.eval()
    
    # ... (rest of the code)
    
    # Convert and Predict
    vec = smiles_to_vector(smiles_string, dim=dim).to(device)
    with torch.no_grad():
        output = model(vec.unsqueeze(0)).mean()
        prob = torch.sigmoid(output).item()
    
    print(f"\n🧪 Testing Drug: {name}")
    print(f"SMILES: {smiles_string}")
    print(f"Prediction: {'PASSED ✅' if prob > 0.5 else 'BLOCKED ❌'} ({prob:.2%})")

if __name__ == "__main__":
    # Caffeine (Should PASS)
    run_test("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "Caffeine")
    # Dopamine (Should typically be BLOCKED - it needs a precursor like L-Dopa)
    run_test("C1=CC(=C(C=C1CCN)O)O", "Dopamine")