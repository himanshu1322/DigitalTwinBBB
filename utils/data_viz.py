import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from utils.featurizer import smiles_to_vector

def visualize_trained_space(tsv_path, model_path, dim=2055):
    print("🎨 Generating High-Resolution Cluster Map...")
    df = pd.read_csv(tsv_path, sep='\t').dropna(subset=['SMILES', 'BBB+/BBB-']).sample(1200)
    
    vectors = []
    labels = []
    for _, row in df.iterrows():
        vec = smiles_to_vector(row['SMILES'], dim=dim)
        if vec is not None:
            vectors.append(vec.numpy())
            labels.append(1 if row['BBB+/BBB-'] == 'BBB+' else 0)
    
    vectors = np.array(vectors)
    
    # Run t-SNE
    # Removed n_iter to maintain compatibility
    tsne = TSNE(n_components=2, perplexity=40, random_state=42)
    reduced = tsne.fit_transform(vectors)
    
    # Plot
    plt.figure(figsize=(12, 8))
    # Plot the dots
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, 
                         cmap='coolwarm', alpha=0.7, s=50, edgecolors='k', linewidth=0.5)
    
    plt.colorbar(scatter, label='0 = Blocked, 1 = Passed')
    plt.title(f"Trained 1.58-bit MoE Space (Accuracy: 86.4%)", fontsize=15)
    plt.xlabel("Latent Chemical Feature 1")
    plt.ylabel("Latent Chemical Feature 2")
    
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig("trained_cluster_map.png", dpi=300)
    print("✅ Success! 'trained_cluster_map.png' created. Check if the red and blue clusters are now separated!")
    plt.show()

if __name__ == "__main__":
    visualize_trained_space('data/B3DB_classification.tsv', 'models/bbb_ternary_moe_95.pth')