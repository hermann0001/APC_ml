import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import normalize_name
import gc
from MetaSpotifyDataExtractor import get_spotify_metadata
import sys

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

# reorganizing order of columns
dataframe = dataframe.reindex(columns=['playlist_id', 'playlist_name', 'track_id', 'track_name', 'track_uri', 'pos', 'artist_id', 'artist_name', 'artist_uri'])

# Normalize text fields (idk why this makes crash)
dataframe['playlist_name'] = dataframe['playlist_name'].apply(normalize_name)
dataframe['track_name'] = dataframe['track_name'].apply(normalize_name)
dataframe['artist_name'] = dataframe['artist_name'].apply(normalize_name)

# Save the dataframe in high performance dataframe on disk
dataframe.to_feather(SRC_FOLDER + 'dataframe.feather')


######################################################  <--- MOVE THIS PART TO ANOTHER FILE

# # Number of playlists
# num_playlists = dataframe['pid'].nunique()

# # Number of tracks
# num_tracks = dataframe['track_uri'].count()

# # Number of unique tracks
# num_unique_tracks = dataframe['track_uri'].nunique()

# # Number of unique albums
# num_unique_albums = dataframe['album_uri'].nunique()

# # Number of unique artists
# num_unique_artists = dataframe['artist_uri'].nunique()

# # Number of unique playlist titles
# num_unique_playlist_titles = dataframe['name'].nunique()

# # Number of unique normalized playlist titles
# num_unique_normalized_playlist_titles = dataframe['normalized_playlist_title'].nunique()

# # Average playlist length (number of tracks)
# average_playlist_length = dataframe['num_tracks'].mean()

# # Set up the plotting style
# sns.set_theme(style="whitegrid")

# # Data for plotting
# metrics = {
#     'Number of Playlists': num_playlists,
#     'Number of Tracks': num_tracks,
#     'Number of Unique Tracks': num_unique_tracks,
#     'Number of Unique Albums': num_unique_albums,
#     'Number of Unique Artists': num_unique_artists,
#     'Number of Unique Playlist Titles': num_unique_playlist_titles,
#     'Number of Unique Normalized Playlist Titles': num_unique_normalized_playlist_titles,
#     'Average Playlist Length (Number of tracks)': average_playlist_length
# }

# # Convert metrics to DataFrame for plotting
# metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])

# print(metrics_df)
# dataframe = dataframe.drop('name', axis=1)

# # Distribution of number of tracks in playlists
# plt.figure(figsize=(10, 6))
# sns.histplot(dataframe['num_tracks'], bins=50, kde=True)
# plt.title('Distribution of Number of Tracks in Playlists')
# plt.xlabel('Number of Tracks')
# plt.ylabel('Frequency')
# plt.savefig('./tracks_in_playlists.png')  

# # Distribution of number of artists in playlists
# plt.figure(figsize=(10, 6))
# sns.histplot(dataframe['num_artists'], bins=50, kde=True)
# plt.title('Distribution of Number of Artists in Playlists')
# plt.xlabel('Number of Artists')
# plt.ylabel('Frequency')
# plt.savefig('./artists_in_playlists.png')  

# # Un box plot (o box-and-whisker plot) mostra la distribuzione dei dati
# # quantitativi.

# plt.figure(figsize=(10, 6))
# sns.boxplot(data=dataframe, x = dataframe['playlist_duration_mins'])
# plt.xlabel('Playlist duration (sec)')
# plt.savefig('./playlist_duration_boxplot.png')  

# mean = dataframe['playlist_duration_mins'].mean()
# std = dataframe['playlist_duration_mins'].std()
# lower_cut, upper_cut = mean -2 * std , mean +2 * std
# dataframe = dataframe[(dataframe['playlist_duration_mins'] > lower_cut) & (dataframe['playlist_duration_mins'] < upper_cut)]

# plt.figure(figsize=(10, 6))

# plt.subplot(1,3,1)
# sns.violinplot(data=dataframe, x = dataframe['playlist_duration_mins'])
# plt.xlabel('Playlist duration (minutes)')
# plt.title('Violin Plot of Playlist Duration')


# plt.subplot(1,3,2)
# sns.histplot(data=dataframe, x = dataframe['playlist_duration_mins'], bins=50)
# plt.xlabel('Playlist duration (minutes)')
# plt.title('Histogram of Playlist Duration')

# plt.subplot(1,3,3)
# sns.boxplot(data=dataframe, x = dataframe['playlist_duration_mins'])
# plt.xlabel('Playlist duration (minutes)')
# plt.title('Box plot of Playlist Duration')

# plt.savefig('after_playlist_duration_shrink.png')  




# plt.figure(figsize=(10, 6))
# sns.boxplot(data=dataframe, x = dataframe['track_duration_s'])
# plt.xlabel('track duration (sec)')
# plt.savefig('./track_duration_box.png')  

# mean = dataframe['track_duration_s'].mean()
# std = dataframe['track_duration_s'].std()

# print(f"mean: {mean}, std: {std}")
# print(dataframe.shape)
# lower_cut, upper_cut = mean -2 * std , mean +2 * std
# dataframe = dataframe[(dataframe['track_duration_s'] > lower_cut) & (dataframe['track_duration_s'] < upper_cut)]
# print(dataframe.shape)


# dataframe['zscore'] = (dataframe['track_duration_s'] - dataframe['track_duration_s'].mean()) / std
# dataframe = dataframe[(dataframe.zscore > -2) & (dataframe.zscore < 2)]
# print(dataframe.shape)

# plt.figure(figsize=(10, 6))

# plt.subplot(1,3,1)
# sns.violinplot(data=dataframe, x = dataframe['track_duration_s'])
# plt.xlabel('Track duration (seconds)')
# plt.title('Violin Plot of Track Duration')


# plt.subplot(1,3,2)
# sns.histplot(data=dataframe, x = dataframe['track_duration_s'], bins=50)
# plt.xlabel('Track duration (seconds)')
# plt.title('Histogram of Track Duration')

# plt.subplot(1,3,3)
# sns.boxplot(data=dataframe, x = dataframe['track_duration_s'])
# plt.xlabel('Track duration (seconds)')
# plt.title('Box plot of Track Duration')

# plt.savefig('after_Track_duration_shrink.png')  