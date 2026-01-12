# import pandas as pd
# import numpy as np
# import librosa
# import os
# from sklearn.preprocessing import StandardScaler
# from sklearn.feature_extraction.text import TfidfVectorizer

# def extract_mfcc(wav_path, n_mfcc=13):
#     """Extracts mean MFCCs from a raw audio file."""
#     try:
#         y, sr = librosa.load(wav_path, duration=30)
#         mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
#         return np.mean(mfcc.T, axis=0)
#     except:
#         return np.zeros(n_mfcc)

# def load_hybrid_dataset():
#     # 1. Load Audio Features
#     audio_df = pd.read_csv('data/features_30_sec.csv')
    
#     # 2. Load Bangla Lyrics
#     lyrics_df = pd.read_csv('data/lyrics/BanglaSongLyrics.csv')
    
#     # NLP: Vectorize Bangla Lyrics (Hard Task: Multi-modal)
#     tfidf = TfidfVectorizer(max_features=100)
#     lyrics_features = tfidf.fit_transform(lyrics_df['lyrics'].fillna('')).toarray()
    
#     # For this project, we will combine audio features and lyrics 
#     # (Since IDs might not match perfectly, we simulate a hybrid set of size N)
#     min_len = min(len(audio_df), len(lyrics_df))
#     audio_feats = audio_df.drop(columns=['filename', 'length', 'label']).values[:min_len]
#     lyrics_feats = lyrics_features[:min_len]
#     labels = audio_df['label'].values[:min_len]

#     # Combine: Audio (58) + Lyrics (100) = 158 features
#     hybrid_features = np.hstack((audio_feats, lyrics_feats))
    
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(hybrid_features)
    
#     return X_scaled, labels, hybrid_features.shape[1]


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

def load_multi_modal_data():
    # 1. Load Audio (1000 rows)
    audio_df = pd.read_csv('data/features_30_sec.csv')
    labels = audio_df['label']
    audio_numeric = audio_df.drop(columns=['filename', 'length', 'label'])
    
    # 2. Load Lyrics (Bangla)
    lyrics_df = pd.read_csv('data/lyrics/BanglaSongLyrics.csv')
    
    # Vectorize Bangla lyrics into 100 features
    tfidf = TfidfVectorizer(max_features=100)
    lyrics_vec = tfidf.fit_transform(lyrics_df['lyrics'].fillna('')).toarray()
    
    # 3. Align (Take the first 1000 rows of lyrics to match audio)
    lyrics_features = lyrics_vec[:len(audio_df)]
    audio_features = audio_numeric.values
    
    # 4. Fusion (Horizontal Concatenation)
    combined_features = np.hstack((audio_features, lyrics_features))
    
    # 5. Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(combined_features)
    
    return X_scaled, labels.values, combined_features.shape[1]