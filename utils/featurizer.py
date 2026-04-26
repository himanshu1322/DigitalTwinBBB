from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator # New module
from rdkit.Chem import Descriptors
import torch
import numpy as np
import math

# Initialize the generator ONCE outside the function to save memory
# radius=2 is standard for Morgan Fingerprints
fm_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def smiles_to_vector(smiles, dim=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    
    # 1. Existing 2048-bit Fingerprint
    fp = fm_gen.GetFingerprint(mol)
    fp_array = np.zeros((0,), dtype=np.float32)
    Chem.DataStructs.ConvertToNumpyArray(fp, fp_array)
    
    # Inside smiles_to_vector function:
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    mw = Descriptors.MolWt(mol)
    hbd = Descriptors.NumHDonors(mol)     # Stickiness (Crucial for Dopamine)
    hba = Descriptors.NumHAcceptors(mol)  # Polarity
    charge = Chem.GetFormalCharge(mol)
    heavy_atoms = mol.GetNumHeavyAtoms()


    # Use this bio_features block for stability
    hbd_penalty = 1 / (1 + math.exp(-(hbd - 2.5))) # Sigmoid: 0-2 HBD = low, 3+ HBD = high

    bio_features = np.array([
        (logp + 5) / 15,    
        tpsa / 150,         
        mw / 600,           
        hbd_penalty,        # This is the "Switch": 0 for Caffeine, 1 for Dopamine
        hba / 8,            
        (charge + 1) / 2,   
        heavy_atoms / 40    
    ], dtype=np.float32)

    # Combined vector is now 2055
    combined_vec = np.concatenate([fp_array, bio_features])
    return torch.tensor(combined_vec).float()