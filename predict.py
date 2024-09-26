import pandas as pd
import numpy as np
from gensim.models import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
import sys

MDL_FOLDER = 'models/'

model = Doc2Vec.load(MDL_FOLDER + "d2v-trained-model.model")

def reccomend_songs_doc2vec(playlist_id, model, top_n=10): 
    """Generate recommendations based on Doc2Vec model."""
    inferred_vector = model.infer_vector(playlist_id)

    # Inizializza le somiglianze
    similar_playlists = model.dv.most_similar([inferred_vector], topn=top_n)

    recommended_tracks = []
    for playlist_id, simalirity in similar_playlists:
        playlist_tracks = ...

    # Calcola le similarità con tutte le tracce nel dataset
    for idx, row in df.iterrows():
        # Crea il documento per la traccia
        track_doc = [str(row['artist_name']), str(row['track_name'])]
        # Inferisci il vettore per la traccia
        track_vector = model.infer_vector(track_doc)
        # Calcola la similarità
        similarity = cosine_similarity([playlist_vector], [track_vector])[0][0]
        similarities.append((row['track_name'], row['artist_name'], similarity))

    # Ordina le tracce in base alla similarità
    similarities.sort(key=lambda x: x[2], reverse=True)

    # Restituisci i top N brani raccomandati
    return similarities[:top_n]

# Esempio di utilizzo
recommended_songs = recommend_songs_for_playlist("Brasileiras", unique_playlists, model, top_n=10)
for song in recommended_songs:
    print(f"Track: {song[0]}, Artist: {song[1]}, Similarity: {song[2]:.4f}")