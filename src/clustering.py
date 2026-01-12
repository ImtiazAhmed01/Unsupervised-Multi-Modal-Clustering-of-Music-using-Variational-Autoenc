import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, calinski_harabasz_score, 
                             davies_bouldin_score, adjusted_rand_score, 
                             normalized_mutual_info_score)

def calculate_purity(y_true, y_pred):
    contingency_matrix = pd.crosstab(y_true, y_pred)
    return np.sum(contingency_matrix.max(axis=0)) / np.sum(contingency_matrix.values)

def run_advanced_clustering(latent_space, true_labels):
    print("Executing K-Means Clustering...")
    n_clusters = len(np.unique(true_labels))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    pred_labels = kmeans.fit_predict(latent_space)
    
    # Calculate all metrics requested in the rubric
    metrics = {
        "Silhouette": silhouette_score(latent_space, pred_labels),
        "CH_Index": calinski_harabasz_score(latent_space, pred_labels),
        "DB_Index": davies_bouldin_score(latent_space, pred_labels),
        "ARI": adjusted_rand_score(true_labels, pred_labels),
        "NMI": normalized_mutual_info_score(true_labels, pred_labels),
        "Purity": calculate_purity(true_labels, pred_labels)
    }
    
    # Save to CSV
    df_metrics = pd.DataFrame([metrics])
    df_metrics.to_csv('results/clustering_metrics.csv', index=False)
    print("Metrics saved to results/clustering_metrics.csv")
    return pred_labels