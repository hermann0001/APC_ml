import pandas as pd
import numpy as np
from gensim.models import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import sys, os

MDL_FOLDER = 'models/'
CHALLENGE_FOLDER = 'formatted/challenge_set/'
DATASET_FOLDER = 'formatted/dataset/'
PLAYLIST_TRACKS_CSV = 'playlists_tracks.csv'
PLAYLISTS_CSV = 'playlists.csv'

# Assume `model` is your trained Doc2Vec model
# Assume `test_playlists` is a list of playlists for testing with their ground truth
# Each entry should have a 'playlist_id' and 'ground_truth' list of track URIs

def get_recommendations(playlist_name, model, top_n=10):
    # Retrieve similar playlists and get their recommended songs
    similar_playlists = model.dv.most_similar(playlist_name, topn=top_n)
    recommended_songs = []
    
    for pl_name, _ in similar_playlists:
        # Assuming you have a function to get songs from a playlist name
        songs = get_songs(pl_name)
        recommended_songs.extend(songs)
    
    return list(set(recommended_songs))  # Return unique songs

def calculate_metrics(test_playlists, model, top_n=10):
    correct_recommendations = 0
    total_recommendations = 0
    relevant_recommendations = 0

    for pl in test_playlists:
        ground_truth = set(pl['ground_truth'])
        recommendations = get_recommendations(pl['name'].lower(), model, top_n)

        # Update counts
        total_recommendations += len(recommendations)
        correct_recommendations += len(ground_truth.intersection(recommendations))
        relevant_recommendations += len(recommendations)

    accuracy = correct_recommendations / total_recommendations if total_recommendations > 0 else 0
    precision = correct_recommendations / relevant_recommendations if relevant_recommendations > 0 else 0
    
    return accuracy, precision

# Example usage
test_playlists = [
    {'name': 'My Playlist 1', 'ground_truth': ['track_uri1', 'track_uri2', 'track_uri3']},
    {'name': 'My Playlist 2', 'ground_truth': ['track_uri4', 'track_uri5']},
    # Add more test playlists...
]

def load_test_data():
    #playlist_tracks_test = pd.read_csv(CHALLENGE_FOLDER + PLAYLIST_TRACKS_CSV)
    playlists_test = pd.read_csv(CHALLENGE_FOLDER + PLAYLISTS_CSV, usecols=['playlist_id', 'name',])
    playlists_train = pd.read_csv(DATASET_FOLDER + PLAYLISTS_CSV, usecols=[''])

    print(playlists_train.head())
    print(playlists_train.info())
    sys.exit(1)
    common_ids = playlists_test['playlist_id'].isin(playlists_df['playlist_id']).sum()
    total_ids = len(test_set)
    print(f"{common_ids} playlist_id su {total_ids} trovati in playlists_df")

    print(f"playlists_df duplicated rows: {playlists_df.duplicated().sum()}")


    merged_df = pd.merge(playlists_df,test_set, on='playlist_id', how='left')

    print(merged_df.head())
    
    

if __name__ == '__main__':

    load_test_data()
    model = Doc2Vec.load(MDL_FOLDER + ' d2v-trained-model.model')
    print(f"Model hyperparameters:{model}")
    

    accuracy, precision = calculate_metrics(test_playlists, model, top_n=10)
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
