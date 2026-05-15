# Unsupervised Multi-Modal Clustering of Music using Variational Autoencoders

This project implements an unsupervised learning framework to cluster music data by extracting features from multiple modalities (e.g., audio signals and metadata/text) using **Variational Autoencoders (VAEs)**.

## 🚀 Overview
The goal of this project is to group similar music tracks without manual labels. By using a VAE, we compress high-dimensional multi-modal data into a lower-dimensional latent space, where clustering algorithms (like K-Means or GMM) can more effectively identify patterns.

## 📂 Project Structure
* `src/`: Core logic and VAE model architecture.
* `notebooks/`: Jupyter notebooks for data exploration and visualization.
* `data/`: (Local only) Raw and processed music datasets.
* `results/`: Saved plots, cluster visualizations, and model weights.
* `run_project.py`: The main entry point to train the model and run clustering.

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone 
   cd Unsupervised-Multi-Modal-Clustering-of-Music-using--Variational-Autoencoders
