import os 
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

SRC_FOLDER = 'formatted/dataset/'
SAMPLE = 1000

if __name__ == '__main__':
    artists = pd.read_csv(SRC_FOLDER + 'artists.csv', nrows=SAMPLE)
    tracks =  pd.read_csv(SRC_FOLDER + 'tracks.csv', nrows=SAMPLE)
    playlists =  pd.read_csv(SRC_FOLDER + 'playlists.csv', nrows=SAMPLE)
    playlists_tracks = pd.read_csv(SRC_FOLDER + 'playlist_tracks.csv', nrows=SAMPLE)

    print(artists.head())
    print(artists.info())
    print("\n\n")
    print(tracks.head())
    print(tracks.info())
    print("\n\n")
    print(playlists.head())
    print(playlists.info())
    print("\n\n")
    print(playlists_tracks.head())
    print(playlists_tracks.info())

    interaction_matrix = playlists_tracks.pivot_table(index='playlist_id', columns='track_id', values='pos', aggfunc='mean', fill_value=0)
    user_similarity = cosine_similarity(interaction_matrix)
    print(user_similarity)