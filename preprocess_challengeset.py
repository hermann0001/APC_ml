import pandas as pd
import matplotlib.pyplot as plt
from utils import normalize_name
from MetaSpotifyDataExtractor import get_spotify_metadata

SAMPLE_SIZE = 300000 # no. of rows loaded (total = 1M)
SRC_FOLDER = "formatted/challenge_set/"
PLAYLIST_CSV = SRC_FOLDER + "playlists.csv"
PLAYLIST_TRACKS_CSV = SRC_FOLDER + "playlist_tracks.csv"
    
playlists_df = pd.read_csv(PLAYLIST_CSV, usecols=['playlist_id', 'name'])
playlist_tracks_df = pd.read_csv(PLAYLIST_TRACKS_CSV, usecols=['playlist_id', 'track_id', 'pos'])

# check duplicates in csv, todo: fix generation of csv
print(playlist_tracks_df.duplicated().sum()) 
print(playlists_df.duplicated().sum()) 

playlists_df = playlists_df.rename(columns={'name' : 'playlist_name'})

playlists_df['playlist_id'] = pd.to_numeric(playlists_df['playlist_id'], downcast='integer')
playlist_tracks_df['track_id'] = pd.to_numeric(playlist_tracks_df['track_id'], downcast='integer')
playlist_tracks_df['pos'] = pd.to_numeric(playlist_tracks_df['pos'], downcast='integer')
playlist_tracks_df['playlist_id'] = pd.to_numeric(playlist_tracks_df['playlist_id'], downcast='integer')

dataframe = pd.merge(playlist_tracks_df, playlists_df, on='playlist_id')

# Normalize text fields (idk why this makes crash)
dataframe['playlist_name'] = dataframe['playlist_name'].apply(normalize_name)

# Save the dataframe in high performance dataframe on disk
dataframe.to_feather(SRC_FOLDER + 'dataframe.feather')
