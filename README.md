# 🎵 Unsupervised Multi-Modal Clustering of Music using Variational Autoencoder

An advanced Deep Learning and Music Information Retrieval (MIR) project focused on **unsupervised music clustering** using **Variational Autoencoders (VAE)** and multi-modal feature learning. This project explores how latent representations extracted from audio features can automatically group similar music tracks without requiring labeled data.

---

## 🌟 Project Overview

Music contains rich hidden structures such as rhythm, timbre, melody, harmony, and spectral patterns. Traditional classification approaches rely heavily on labeled datasets, but this project explores an **unsupervised learning approach** where the model learns meaningful representations directly from the music data.

Using a **Variational Autoencoder (VAE)** architecture, the system compresses high-dimensional music features into a low-dimensional latent space and performs clustering to identify hidden relationships between songs and audio patterns.

The project combines concepts from:

* 🎼 Music Information Retrieval (MIR)
* 🤖 Deep Learning
* 🧠 Representation Learning
* 📊 Unsupervised Clustering
* 🎧 Audio Signal Processing

Variational Autoencoders are widely used for representation learning and generative modeling in machine learning. ([Wikipedia][1])

---

# 🚀 Features

* 🎵 Multi-modal music feature extraction
* 🧠 Variational Autoencoder (VAE) based representation learning
* 📊 Unsupervised clustering of music tracks
* 📉 Dimensionality reduction in latent space
* 🎼 Automatic grouping of similar music
* 📈 Visualization of latent embeddings
* 🔍 Audio feature preprocessing pipeline
* 🎧 Spectrogram and feature analysis
* 🤖 Deep unsupervised learning workflow

---

# 🧠 What is a Variational Autoencoder?

A **Variational Autoencoder (VAE)** is a deep generative model that learns compressed latent representations of data while maintaining meaningful structure in the latent space. Unlike traditional autoencoders, VAEs learn probability distributions rather than fixed encodings. ([Wikipedia][1])

Inline mathematical representation of a VAE latent distribution:

q_\phi(z|x)=\mathcal{N}(\mu,\sigma^2 I)

The VAE framework is highly effective for:

* Feature learning
* Clustering
* Music generation
* Representation learning
* Audio analysis

MusicVAE architectures specifically target long music sequence representation learning. ([GitHub][2])

---

# 🎼 Multi-Modal Learning

This project uses **multi-modal features**, meaning multiple representations of music are combined to improve clustering quality.

Possible modalities include:

| Modality             | Description                               |
| -------------------- | ----------------------------------------- |
| 🎵 MFCC Features     | Mel-frequency cepstral coefficients       |
| 📈 Spectrograms      | Time-frequency audio representation       |
| 🔊 Chroma Features   | Harmonic and pitch-related representation |
| 🎼 Tempo Features    | Rhythm and beat information               |
| 🎧 Spectral Features | Timbre and frequency characteristics      |

Multi-modal VAEs improve latent representation learning across complex data sources. ([Yuge (Jimmy) Shi][3])

---

# 🏗️ Project Workflow

```text
Raw Audio Files
        ↓
Audio Preprocessing
        ↓
Feature Extraction
        ↓
Multi-Modal Representation
        ↓
Variational Autoencoder
        ↓
Latent Space Embedding
        ↓
Unsupervised Clustering
        ↓
Visualization & Analysis
```

---

# 📂 Dataset

This project may use publicly available music datasets such as:

## 🎵 Suggested Datasets

### GTZAN Genre Dataset

A widely used music genre classification dataset.

🔗 Dataset Link:
[https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

---

### FMA (Free Music Archive) Dataset

Large-scale open music dataset for MIR tasks.

🔗 Dataset Link:
[https://github.com/mdeff/fma](https://github.com/mdeff/fma)

---

### MAESTRO Dataset

Piano performance dataset for advanced music modeling.

🔗 Dataset Link:
[https://magenta.tensorflow.org/datasets/maestro](https://magenta.tensorflow.org/datasets/maestro)

---

# 🛠️ Technologies Used

## 👨‍💻 Programming Language

* Python

---

## 📚 Libraries & Frameworks

| Library              | Purpose              |
| -------------------- | -------------------- |
| TensorFlow / PyTorch | Deep Learning        |
| Librosa              | Audio Processing     |
| NumPy                | Numerical Computing  |
| Pandas               | Data Handling        |
| Matplotlib           | Visualization        |
| Scikit-learn         | Clustering & Metrics |
| Seaborn              | Data Visualization   |

Librosa is commonly used for audio and music signal analysis in Python. ([GitHub][2])

---

# 🧠 Machine Learning Architecture

## Encoder

Compresses audio features into latent vectors.

## Latent Space

Learns meaningful compressed representations of music.

Example latent variable representation:

z \sim \mathcal{N}(\mu,\sigma^2)

## Decoder

Reconstructs the original music feature representation from latent embeddings.

---

# 📊 Clustering Techniques

Possible clustering methods:

* K-Means
* Gaussian Mixture Models (GMM)
* DBSCAN
* Hierarchical Clustering

Gaussian Mixture approaches are often combined with VAEs for unsupervised clustering tasks. ([HyperAI][4])

---

# 📈 Visualization

The project may visualize:

* 🎵 Latent space embeddings
* 📊 Cluster distributions
* 📉 PCA projections
* 🌌 t-SNE visualizations
* 🎧 Spectrogram comparisons

---

# 🚀 Installation

## 📥 Clone the Repository

```bash
git clone https://github.com/ImtiazAhmed01/Unsupervised-Multi-Modal-Clustering-of-Music-using-Variational-Autoenc.git
cd Unsupervised-Multi-Modal-Clustering-of-Music-using-Variational-Autoenc
```

---

## 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

```bash
python train.py
```

or launch notebook:

```bash
jupyter notebook
```

---

# 📁 Suggested Project Structure

```text
project/
│
├── notebooks/
│   └── exploratory.ipynb
│
├── results/
│   ├── latent_visualization/
│   └── clustering_metrics.csv
│
├── src/
│   ├── __init__.py
│   ├── clustering.py
│   ├── dataset.py
│   ├── evaluation.py
│   └── vae.py
│
├── .gitignore
├── README.md
├── requirements.txt
└── run_project.py
```

---

# 📊 Expected Outcomes

✅ Learn meaningful latent music representations

✅ Automatically cluster similar songs

✅ Discover hidden audio patterns

✅ Reduce high-dimensional feature complexity

✅ Demonstrate unsupervised deep learning for music analysis

---

# 🔬 Research & Inspiration

This project is inspired by research on:

* MusicVAE architectures
* Multi-modal variational learning
* Deep clustering
* Generative audio modeling
* Latent space representation learning

Relevant works and implementations:

* MusicVAE implementations ([GitHub][2])
* Gaussian Mixture VAE clustering ([GitHub][5])
* Variational Deep Embedding (VaDE) ([paperswithcode.com][6])

---

# 🔮 Future Improvements

* 🎼 Music generation from latent space
* 🤖 Transformer-based music modeling
* ☁️ Web deployment
* 🎧 Real-time music recommendation
* 📱 Interactive clustering dashboard
* 🧠 Contrastive representation learning
* 🎵 Genre-independent representation learning

---

# 👨‍💻 Author

## Imtiaz Ahmed

* 🌐 GitHub: [ImtiazAhmed01 GitHub](https://github.com/ImtiazAhmed01?utm_source=chatgpt.com)
* 💼 LinkedIn: [Imtiaz Ahmed LinkedIn](https://www.linkedin.com/in/imtiaz-ahmed-ar/?utm_source=chatgpt.com)

---

# ⭐ Support

If you found this project useful:

⭐ Star the repository

🍴 Fork the project

🛠️ Contribute improvements

---

# 📜 License

This project is licensed under the MIT License.

---

---

# <img width="26" height="26" alt="image" src="https://github.com/user-attachments/assets/921d5c4e-a2bd-488f-9c3e-6403366e408c" /> Completion

This project has been completed on 25/01/2026.

---

# 🏷️ Tags

`Deep Learning` `Variational Autoencoder` `Music Clustering` `Audio Processing` `Machine Learning` `Unsupervised Learning` `Music Information Retrieval` `Python` `AI` `Latent Space`

[1]: https://en.wikipedia.org/wiki/Variational_autoencoder?utm_source=chatgpt.com "Variational autoencoder"
[2]: https://github.com/Variational-Autoencoder/MusicVAE?utm_source=chatgpt.com "GitHub - Variational-Autoencoder/MusicVAE"
[3]: https://yugeten.github.io/publication/2019-06-01-multimodal-VAE?utm_source=chatgpt.com "Variational Mixture-of-Experts Autoencoders for Multi-Modal Deep Generative Models - Yuge (Jimmy) Shi"
[4]: https://hyper.ai/en/papers/1611.02648?utm_source=chatgpt.com "Deep Unsupervised Clustering with Gaussian Mixture Variational Autoencoders | Papers | HyperAI"
[5]: https://github.com/jariasf/GMVAE?utm_source=chatgpt.com "GitHub - jariasf/GMVAE: Implementation of Gaussian Mixture Variational Autoencoder (GMVAE) for Unsupervised Clustering"
[6]: https://paperswithcode.com/paper/variational-deep-embedding-an-unsupervised?utm_source=chatgpt.com "Variational Deep Embedding: An Unsupervised and Generative Approach to Clustering | Papers With Code"
