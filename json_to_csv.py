import os
import pandas as pd
import json
import gc
from tqdm import tqdm
import time

PATH = "archive/data"
PLAYLIST_CSV = "temp/playlists.csv"
TRACK_CSV = "temp/tracks.csv"
BATCH_SIZE = 50                             # number of files loaded per batch
filenames = os.listdir(PATH)

def process_dataset_batch(file_batch):
    batch_dfs = []

    for filename in file_batch:
        fullpath = os.sep.join((PATH,filename))

        with open(fullpath, 'r') as f_in:
            json_data = f_in.read()

        # Read playlists data
        mpd_slice = json.loads(json_data)
        playlists = mpd_slice['playlists']
        mpd_slice_df = pd.json_normalize(playlists)

        batch_dfs.append(mpd_slice_df)
    
    combined_batch_df = pd.concat(batch_dfs, ignore_index=True)

    return combined_batch_df, extract_track_df(combined_batch_df)

def extract_track_df(playlist_df):
    tracks_list = []

    # Iterate over each playlist to extract and flatten the tracks
    for index, row in playlist_df.iterrows():
        tracks = row['tracks']  # Extract the nested 'tracks'
        for track in tracks:
            track['pid'] = row['pid']  # Add playlist ID if needed
            tracks_list.append(track)
    
    # Create a DataFrame from the flattened track data
    tracks_df = pd.DataFrame(tracks_list)

    return tracks_df

def load_data():
    with open(PLAYLIST_CSV, 'w') as p_out, open(TRACK_CSV, 'w') as t_out:
        with tqdm(total=len(filenames) // BATCH_SIZE + 1, desc="Processing files") as pbar:
            for i in range(0, len(filenames), BATCH_SIZE):
                file_batch = filenames[i:i + BATCH_SIZE]
                playlists_batch_df, tracks_batch_df = process_dataset_batch(file_batch)
                playlists_batch_df = playlists_batch_df.drop('tracks', axis=1)

                playlists_batch_df.to_csv(p_out, header=p_out.tell() == 0, index=False)

                tracks_batch_df.to_csv(t_out, header=t_out.tell() == 0, index=False)

                pbar.update(1)

                del playlists_batch_df, tracks_batch_df
                gc.collect()
            time.sleep(1)

if __name__ == "__main__":
    print(f"Ci sono {len(filenames)} files, per un totale di {len(filenames)*1000} playlists, quante ne vuoi caricare?")
    n = int(input())
    del filenames[n:]
    print(f"{len(filenames)} files saranno caricati ({len(filenames)*1000} playlists)")
    load_data()
