import os
import sys
import numpy as np
import tensorflow as tf

# Add src to path
sys.path.append(os.path.abspath("src"))

from dataset import load_multi_modal_data
from vae import BetaVAE
from clustering import run_advanced_clustering
from evaluation import visualize_results

def main():
    print("--- STARTING HARD TASK: MULTI-MODAL BETA-VAE ---")
    os.makedirs('results/latent_visualization', exist_ok=True)

    # 1. Load Data
    X, labels, input_dim = load_multi_modal_data()
    
    # 2. Train Beta-VAE
    vae = BetaVAE(input_dim=input_dim, latent_dim=16, beta=4.0)
    vae.compile(optimizer=tf.keras.optimizers.Adam(0.001))
    vae.fit(X, epochs=50, batch_size=32, verbose=1)

    # 3. Get Latent Space
    z_mean, _, _ = vae.encoder.predict(X)
    
    # 4. Clustering & Metrics
    pred_labels = run_advanced_clustering(z_mean, labels)
    
    # 5. Visualization
    visualize_results(z_mean, labels)
    
    print("\n--- ALL TASKS COMPLETE ---")
    print("Check 'results/clustering_metrics.csv' for the NMI, ARI, and Purity scores.")

if __name__ == "__main__":
    main()