'''
This python code reads the dataset contained in SRC_FOLDER which is a pure json formatted dataset and do various operations. 
The dataset is exploded in plain structures "artists, tracks, playlists, playlists_tracks" and then wrote to respective CSV file in TARGET_FOLDER.
Since the dataset is 32+GB, the operation is executed per batch, it means subgroups of files are loaded togheter and for each batch a write to disk happens.
'''

import os
import pandas as pd
import json
from collections import defaultdict
from utils import *
# folders
SRC_FOLDER = "dataset/data/"
TARGET_FODLER = "formatted/dataset/"
# files
PLAYLISTS_FILE = TARGET_FODLER + "playlists.csv"
TRACKS_FILE = TARGET_FODLER + "tracks.csv"
ARTISTS_FILE = TARGET_FODLER + "artists.csv"
PLAYLISTS_TRACKS_FILE = TARGET_FODLER + "playlist_tracks.csv"

BATCH_SIZE = 100                             # number of files loaded per batch

filenames = os.listdir(SRC_FOLDER)

artist_id_map = defaultdict(lambda: None)
track_id_map = defaultdict(lambda: None)
artist_id_counter = [0]                     # Using list to maintain a mutable counter
track_id_counter = [0]

def getid(uri, uri_map, counter):
    """
    Generates a unique integer ID for a given URI.
    
    Args:
    - uri (str): The URI to process.
    - uri_map (dict): A mapping of URIs to unique integer IDs.
    - counter (list): A list with a single integer representing the next ID to assign.
    
    Returns:
    - int: The unique ID for the URI.
    """
    # Check if URI already has an ID
    if uri_map[uri] is not None:
        return uri_map[uri]
    
    # Generate new ID
    new_id = counter[0] + 1
    counter[0] = new_id

    # Store the new ID in the mapping
    uri_map[uri] = new_id

    return new_id

def process_dataset_batch(file_batch):
    playlists_list = []
    artists_list = []
    tracks_list = []
    playlists_tracks_list = []

    for filename in file_batch:
        print(f"Reading {filename}")
        fullpath = os.sep.join((SRC_FOLDER,filename))

        with open(fullpath, 'r') as f_in:
            json_data = f_in.read()

        # Read playlists data
        mpd_slice = json.loads(json_data)
        playlists = mpd_slice['playlists']
        mpd_slice_df = pd.json_normalize(playlists)

        # Extract and vectorize the main playlist fields
        playlists_df = mpd_slice_df[['pid', 'name', 'num_tracks', 'num_artists', 'num_albums', 'num_followers', 'num_edits', 'duration_ms', 'modified_at', 'collaborative', 'description']].copy()
        playlists_df.columns = [
            'playlist_id', 'name', 'num_tracks', 'num_artists', 'num_albums',
            'num_followers', 'num_edits', 'duration_ms', 'modified_at',
            'collaborative', 'description'
            ]
        
        #Vectorized operation for playlists tracks using `explode`
        df_exploded = mpd_slice_df[['pid', 'tracks']].explode('tracks').reset_index(drop=True)

        df_exploded['artist_uri'] = df_exploded['tracks'].apply(lambda x: x['artist_uri'])
        df_exploded['artist_name'] = df_exploded['tracks'].apply(lambda x: x['artist_name'])
        df_exploded['track_uri'] = df_exploded['tracks'].apply(lambda x: x['track_uri'])
        df_exploded['track_name'] = df_exploded['tracks'].apply(lambda x: x['track_name'])
        df_exploded['album_uri'] = df_exploded['tracks'].apply(lambda x: x['album_uri'])
        df_exploded['duration_ms_track'] = df_exploded['tracks'].apply(lambda x: x['duration_ms'])
        df_exploded['album_name'] = df_exploded['tracks'].apply(lambda x: x['album_name'])
        df_exploded['pos'] = df_exploded['tracks'].apply(lambda x: x['pos'])

        # Deduplicate and generate IDs for artists and tracks
        df_exploded['artist_id'] = df_exploded['artist_uri'].map(lambda uri: getid(uri, artist_id_map, artist_id_counter))
        df_exploded['track_id'] = df_exploded['track_uri'].map(lambda uri: getid(uri, track_id_map, track_id_counter))

        # Simplify URIs
        df_exploded['artist_uri'] = df_exploded['artist_uri'].apply(extract_id_from_uri)
        df_exploded['track_uri'] = df_exploded['track_uri'].apply(extract_id_from_uri)
        df_exploded['album_uri'] = df_exploded['album_uri'].apply(extract_id_from_uri)

        # Create Dataframes
        artists_df = df_exploded[['artist_id', 'artist_uri', 'artist_name']].drop_duplicates().reset_index(drop=True)
        tracks_df = df_exploded[['track_id', 'track_uri', 'track_name', 'artist_id', 'album_uri', 'duration_ms_track', 'album_name']].drop_duplicates().reset_index(drop=True)
        playlists_tracks_df = df_exploded[[
            'pid', 'track_id', 'artist_id', 'pos'
        ]].rename(columns={'pid': 'playlist_id'}).reset_index(drop=True)

        # Append data to lists
        playlists_list.append(playlists_df)
        artists_list.append(artists_df)
        tracks_list.append(tracks_df)
        playlists_tracks_list.append(playlists_tracks_df)

    # Concatenate all data
    combined_playlists_df = pd.concat(playlists_list, ignore_index=True)
    combined_artists_df = pd.concat(artists_list, ignore_index=True).drop_duplicates().reset_index(drop=True)
    combined_tracks_df = pd.concat(tracks_list, ignore_index=True).drop_duplicates().reset_index(drop=True)
    combined_playlists_tracks_df = pd.concat(playlists_tracks_list, ignore_index=True)

    return combined_playlists_df, combined_artists_df, combined_tracks_df, combined_playlists_tracks_df

if __name__ == "__main__":
    print(f"Ci sono {len(filenames)} files, per un totale di {len(filenames)*1000} playlists, quante ne vuoi caricare?")
    n = int(input())
    del filenames[n:]
    print(f"{n} files saranno caricati ({n*1000} playlists)")
    
    with open(PLAYLISTS_FILE, 'w') as p_out, open(TRACKS_FILE, 'w') as t_out, open(ARTISTS_FILE, 'w') as a_out, open(PLAYLISTS_TRACKS_FILE, 'w') as pt_out:
        for i in range(0, len(filenames), BATCH_SIZE):
            file_batch = filenames[i:i + BATCH_SIZE]

            playlists_batch_df, artists_batch_df, tracks_batch_df, playlists_tracks_batch_df = process_dataset_batch(file_batch)

            playlists_batch_df.to_csv(p_out, header=p_out.tell() == 0, index=False)
            artists_batch_df.to_csv(a_out, header=a_out.tell() == 0, index=False)
            tracks_batch_df.to_csv(t_out, header=t_out.tell() == 0, index=False)
            playlists_tracks_batch_df.to_csv(pt_out, header=pt_out.tell() == 0, index=False)
            print(f"Wrote {i + BATCH_SIZE} files to .csv")
    
    check_csv(PLAYLISTS_FILE, n*1000, id_column="playlist_id")

    
