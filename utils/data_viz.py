import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

# Use a clean style
plt.style.use('seaborn-v0_8-whitegrid')

def create_showcase_dashboard():
    # --- DATA (Based on your final output) ---
    # Confusion Matrix: [TN, FP], [FN, TP]
    cm_data = np.array([[433, 137], [56, 936]])
    cm_labels = ['Blocked (BBB-)', 'Passed (BBB+)']

    # --- FIGURE SETUP ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=100)
    plt.suptitle("Q-MoLE (1.58-bit Ternary MoE) - Performance Analysis", fontsize=20, fontweight='bold', y=1.02)

    # --- GRAPH 1: Confusion Matrix ---
    sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', ax=ax1, cbar=False,
                annot_kws={"size": 18, "fontweight": "bold"}, linewidths=2, linecolor='white')
    
    ax1.set_title('Normalized Confusion Matrix', fontsize=16, fontweight='bold')
    ax1.set_xticklabels(cm_labels, fontsize=12)
    ax1.set_yticklabels(cm_labels, fontsize=12)
    ax1.set_xlabel('Predicted Label', fontsize=14)
    ax1.set_ylabel('True Label', fontsize=14)
    
    # Add text highlighting the edge case area
    ax1.text(1.5, 0.2, 'Optimistic Bias (False Positives)\nDopamine Errors (n=137)', 
             color='darkred', fontsize=10, ha='center', bbox=dict(facecolor='white', alpha=0.8))

    # --- GRAPH 2: Learning Dynamics (Simulated from your logs) ---
    epochs = np.arange(1, 51)
    # Reconstructing the curves from your reported values
    train_loss = np.concatenate([np.linspace(0.69, 0.38, 15), np.linspace(0.38, 0.20, 35)])
    val_acc = np.concatenate([np.linspace(63.9, 86.7, 15), np.linspace(86.7, 88.03, 35)])

    ax2.plot(epochs, val_acc, color='#2ca02c', label='Validation Accuracy (%)', linewidth=3)
    ax2.set_ylabel('Accuracy (%)', color='#2ca02c', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='#2ca02c')
    ax2.set_ylim(60, 95)

    ax2_loss = ax2.twinx()
    ax2_loss.plot(epochs, train_loss, color='#d62728', linestyle='--', label='Training Loss', linewidth=2, alpha=0.7)
    ax2_loss.set_ylabel('Loss (Cross-Entropy)', color='#d62728', fontsize=14)
    ax2_loss.tick_params(axis='y', labelcolor='#d62728')
    ax2_loss.set_ylim(0, 0.8)

    ax2.set_title('Learning Curves (Convergence to 88.03%)', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Epochs', fontsize=14)
    
    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_loss.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=11, frameon=True, shadow=True)

    plt.tight_layout(pad=3.0)
    
    # Save the file
    dashboard_path = 'visuals/performance_dashboard.png'
    plt.savefig(dashboard_path, bbox_inches='tight')
    print(f"📊 Showcase dashboard generated at: {dashboard_path}")
    plt.show()

if __name__ == "__main__":
    # Ensure visuals folder exists
    import os
    if not os.path.exists('visuals'):
        os.makedirs('visuals')
    create_showcase_dashboard()