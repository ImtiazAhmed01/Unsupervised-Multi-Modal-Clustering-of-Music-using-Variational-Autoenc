import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

def visualize_results(latent_space, labels):
    print("Generating t-SNE Plot...")
    tsne = TSNE(n_components=2, random_state=42)
    z_2d = tsne.fit_transform(latent_space)
    
    plt.figure(figsize=(12, 8))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = (labels == label)
        plt.scatter(z_2d[mask, 0], z_2d[mask, 1], label=label, alpha=0.6, s=15)
    
    plt.legend(title="Genres", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title("Beta-VAE Multi-Modal Latent Space (Audio + Bangla Lyrics)")
    plt.tight_layout()
    plt.savefig('results/latent_visualization/tsne_clusters.png')
    print("Visualization saved.")