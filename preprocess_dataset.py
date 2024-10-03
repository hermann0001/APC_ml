import pandas as pd
from utils import normalize_name
import gc
from MetaSpotifyDataExtractor import get_spotify_metadata
import logging
from sklearn.model_selection import train_test_split
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
SAMPLE_SIZE = 100000 # no. of rows loaded (total = 1M)
SRC_FOLDER = "formatted/dataset/"
PLAYLIST_CSV = SRC_FOLDER + "playlists.csv"
TRACK_CSV = SRC_FOLDER + "tracks.csv"
ARTISTS_CSV = SRC_FOLDER + "artists.csv"
PLAYLIST_TRACKS_CSV = SRC_FOLDER + "playlist_tracks.csv"

def fill_nan_names(artist_df, tracks_df):
    logging.info("Retrieving metadata...")
    nan_track_name = tracks_df[tracks_df['track_name'].isna()]
    nan_artist_name = artist_df[artist_df['artist_name'].isna()]

    for index, row in nan_track_name.iterrows():
        metadata = get_spotify_metadata(uri=row['track_uri'], type='track')
        if metadata: 
            tracks_df.at[index, 'track_name'] = metadata['name']
    
    for index, row in nan_artist_name.iterrows():
        metadata = get_spotify_metadata(uri=row['artist_uri'], type='artist')
        if metadata:
            artist_df.at[index, 'artist_name'] = metadata['name']

logging.info("Reading csv...")
artists_df = pd.read_csv(ARTISTS_CSV)
tracks_df = pd.read_csv(TRACK_CSV, usecols=['track_id', 'track_name', 'artist_id', 'track_uri'])
playlists_df = pd.read_csv(PLAYLIST_CSV, usecols=['playlist_id', 'name'], nrows=SAMPLE_SIZE)
playlist_tracks_df = pd.read_csv(PLAYLIST_TRACKS_CSV, usecols=['playlist_id', 'track_id', 'pos'])

# check duplicates in csv, todo: fix generation of csv
logging.info(f"playlist_tracks_df duplicated rows: {playlist_tracks_df.duplicated().sum()}")
logging.info(f"tracks_df duplicated rows: {tracks_df.duplicated().sum()}")
logging.info(f"playlists_df duplicated rows: {playlists_df.duplicated().sum()}")
logging.info(f"artists_df duplicated rows: {artists_df.duplicated().sum()}")

tracks_df = tracks_df.drop_duplicates().reset_index(drop=True)
artists_df = artists_df.drop_duplicates().reset_index(drop=True)

logging.info(f"tracks_df duplicated rows: {tracks_df.duplicated().sum()}")
logging.info(f"artists_df duplicated rows: {artists_df.duplicated().sum()}")

playlists_df = playlists_df.rename(columns={'name': 'playlist_name'})

# count NaN and fill NaN
logging.info(f"NaN in artists_df: {artists_df.isna().sum()}")
logging.info(f"NaN in tracks_df: {tracks_df.isna().sum()}\n")
fill_nan_names(artists_df, tracks_df)
logging.info(f"NaN in artists_df: {artists_df.isna().sum()}")
logging.info(f"NaN in tracks_df: {tracks_df.isna().sum()}")

logging.info("Normalizing text fields...")
playlists_df['playlist_name'] = playlists_df['playlist_name'].apply(normalize_name)
tracks_df['track_name'] = tracks_df['track_name'].apply(normalize_name)
artists_df['artist_name'] = artists_df['artist_name'].apply(normalize_name)

logging.info("Downcasting data types...")
playlists_df['playlist_id'] = pd.to_numeric(playlists_df['playlist_id'], downcast='integer')
tracks_df['artist_id'] = pd.to_numeric(tracks_df['artist_id'], downcast='integer')
tracks_df['track_id'] = pd.to_numeric(tracks_df['track_id'], downcast='integer')
playlist_tracks_df['track_id'] = pd.to_numeric(playlist_tracks_df['track_id'], downcast='integer')
playlist_tracks_df['pos'] = pd.to_numeric(playlist_tracks_df['pos'], downcast='integer')
playlist_tracks_df['playlist_id'] = pd.to_numeric(playlist_tracks_df['playlist_id'], downcast='integer')

logging.info("Merging dataframes...")
dataframe = pd.merge(playlist_tracks_df, playlists_df, on='playlist_id')
dataframe = pd.merge(dataframe, tracks_df, on='track_id')
dataframe = pd.merge(dataframe, artists_df, on='artist_id')

assert dataframe['playlist_id'].nunique() == SAMPLE_SIZE, 1

# reorganizing order of columns
dataframe = dataframe.reindex(columns=['playlist_id', 'playlist_name', 'track_id', 'track_name', 'track_uri', 'pos', 'artist_id', 'artist_name', 'artist_uri'])

# free memory! or at least make sure you do it
del tracks_df
del playlists_df
del playlist_tracks_df
del artists_df
gc.collect()

logging.info("Saving dataframe to disk...")

# Save the dataframe in high performance dataframe on disk
dataframe.to_feather('formatted/dataset/dataframe.feather')

logging.info("Splitting train and test set...")

playlists = []
for playlist_id, group in dataframe.groupby('playlist_id'):
  tracks = group['track_id'].values.tolist()
  playlists.append(tracks)

clean_playlists = [p for p in playlists if len(p) > 1]
logging.info(f"Playlist with at least 1 song: {len(clean_playlists)}")

train, test = train_test_split(clean_playlists, test_size=1000, shuffle=True, random_state=666)

with open(SRC_FOLDER + 'train.pkl', 'wb') as train_f, open(SRC_FOLDER + 'test.pkl', 'wb') as test_f:
    pickle.dump(train, train_f)
    pickle.dump(test, test_f)
