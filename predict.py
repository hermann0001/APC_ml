import pandas as pd
import numpy as np
from gensim.models import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
import sys

MDL_FOLDER = 'models/'

df = pd.read_feather("formatted/dataframe.feather")

unique_playlists = df.drop_duplicates(subset='playlist_id').head(1000)
print(unique_playlists.head())

model = Doc2Vec.load(MDL_FOLDER + "d2v-trained-model.model")

def recommend_songs_for_playlist(playlist_name, unique_playlists ,model, top_n=10):
    # Filtra il DataFrame per ottenere la playlist
    playlist_data = unique_playlists[unique_playlists['playlist_name'] == playlist_name]
    
    # Crea un documento per inferire il vettore della playlist
    playlist_doc = [str(playlist_name)] + playlist_data['artist_name'].tolist() + playlist_data['track_name'].tolist()
    
    # Inferisci il vettore per la playlist
    playlist_vector = model.infer_vector(playlist_doc)

    # Inizializza le somiglianze
    similar_playlists = model.dv.most_similar([playlist_vector], topn=top_n)

    recommended_tracks = []
    for playlist_id, similarity in similar_playlists:
        playlist_tracks = train_pla

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