import pandas as pd
import json
import os
from utils import *

# data folder
META_SRC_FOLDER = "formatted/dataset/"
SRC_FOLDER = "challenge_set/"
TARGET_FOLDER = "formatted/challenge_set/"
# files
TEST_FILE = "challenge_set.json"
PLAYLISTS_FILE = "playlists.csv"
ARTISTS_FILE = "artists.csv"
TRACKS_FILE = "tracks.csv"
PLAYLISTS_TRACKS_FILE = "playlists_tracks.csv"

if __name__ == '__main__':
    artists = pd.read_csv(os.path.join(META_SRC_FOLDER, ARTISTS_FILE))
    tracks = pd.read_csv(os.path.join(META_SRC_FOLDER, TRACKS_FILE))

    # Create mappings for quick lookup
    # Ensure index is unique
    trackmap = tracks[['track_id', 'track_uri']].drop_duplicates().set_index('track_uri')
    artistmap = artists[['artist_id', 'artist_uri']].drop_duplicates().set_index('artist_uri')

    # Verify that the indices are unique
    assert trackmap.index.is_unique, "trackmap index is not unique"
    assert artistmap.index.is_unique, "artistmap index is not unique"

    with open(os.path.join(SRC_FOLDER, TEST_FILE), 'r') as f:
        challenge_set_json = json.load(f)
    
    playlists_challenge = challenge_set_json['playlists']

    # convert the playlists to DataFrame
    playlists_challenge_df = pd.json_normalize(playlists_challenge)
    playlists_df = playlists_challenge_df[['pid', 'name', 'num_tracks', 'num_samples', 'num_holdouts']]
    playlists_df.columns = ['playlist_id', 'name', 'num_tracks', 'num_samples', 'num_holdouts']

    # Explode tracks field
    df_exploded = playlists_challenge_df[['pid', 'tracks']].explode('tracks').reset_index(drop=True)
    df_filtered = df_exploded[pd.notna(df_exploded['tracks'])].copy()
    df_filtered['artist_uri'] = df_filtered['tracks'].apply(lambda x: x['artist_uri'])
    df_filtered['track_uri'] = df_filtered['tracks'].apply(lambda x: x['track_uri'])
    df_filtered['pos'] = df_filtered['tracks'].apply(lambda x: x['pos'])

    # Simplify URIs
    df_filtered['artist_uri'] = df_filtered['artist_uri'].apply(extract_id_from_uri)
    df_filtered['track_uri'] = df_filtered['track_uri'].apply(extract_id_from_uri)
    
    # map URIs to IDs
    df_filtered['artist_id'] = df_filtered['artist_uri'].map(artistmap['artist_id'])
    df_filtered['track_id'] = df_filtered['track_uri'].map(trackmap['track_id'])


    # Create playlists_track DataFrame
    playlists_tracks_df = df_filtered[['pid', 'track_id', 'artist_id', 'pos']].rename(columns={'pid': 'playlist_id'}).reset_index(drop=True)


    # Save csv
    playlists_df.to_csv(os.path.join(TARGET_FOLDER, PLAYLISTS_FILE), index=False)
    playlists_tracks_df.to_csv(os.path.join(TARGET_FOLDER, PLAYLISTS_TRACKS_FILE), index=False)

    # check csv
    check_csv(os.path.join(TARGET_FOLDER, PLAYLISTS_FILE), 10000, id_column="playlist_id")