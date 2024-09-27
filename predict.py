import pandas as pd
import numpy as np
from gensim.models import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import sys, os

MDL_FOLDER = 'models/'
SRC_FOLDER = 'formatted/dataset/'
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

if __name__ == '__main__':
    model = Doc2Vec.load(MDL_FOLDER + 'd2v-trained-model.model')
    test_set = pd.read_feather(SRC_FOLDER + 'test.feather')
    print(f"Model hyperparameters:{model}")
    

    accuracy, precision = calculate_metrics(test_set, model, top_n=10)
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
