import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from .featurizer import smiles_to_vector

class B3DBDataset(Dataset):
    def __init__(self, dataframe, dim=2055):
        self.smiles = dataframe['SMILES'].values
        self.labels = [1 if l == 'BBB+' else 0 for l in dataframe['BBB+/BBB-'].values]
        self.dim = dim

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        vector = smiles_to_vector(self.smiles[idx], dim=self.dim)
        if vector is None: vector = torch.zeros(self.dim)
        return vector, torch.tensor(self.labels[idx], dtype=torch.float32)

def get_stratified_loaders(tsv_path, batch_size=32, dim=2048):
    df = pd.read_csv(tsv_path, sep='\t').dropna(subset=['SMILES', 'BBB+/BBB-'])
    
    # 80/20 Split with Stratification (Keeps the ratio of Pass/Fail the same)
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['BBB+/BBB-']
    )
    
    train_loader = DataLoader(B3DBDataset(train_df, dim), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(B3DBDataset(test_df, dim), batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader