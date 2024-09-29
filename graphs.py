import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import normalize_name
import gc
from MetaSpotifyDataExtractor import get_spotify_metadata
from sklearn.model_selection import train_test_split


dataframe = pd.read_feather('formatted/dataframe.feather')
tracks = pd.read_csv('formatted/dataset/tracks.csv')
playlists = pd.read_csv('formatted/dataset/playlists.csv')

print(dataframe.columns)


# Number of playlists
num_playlists = dataframe['playlist_id'].nunique()

# Number of tracks
num_tracks = dataframe['track_uri'].count()

# # Number of unique tracks
num_unique_tracks = dataframe['track_uri'].nunique()

# Number of unique albums
num_unique_albums = tracks['album_uri'].nunique()

# Number of unique artists
num_unique_artists = dataframe['artist_uri'].nunique()

# Number of unique playlist titles
num_unique_playlist_titles = dataframe['playlist_name'].nunique()

# Number of unique normalized playlist titles
#num_unique_normalized_playlist_titles = dataframe['normalized_playlist_title'].nunique()

# Average playlist length (number of tracks)
average_playlist_length = playlists['num_tracks'].mean()

# Set up the plotting style
sns.set_theme(style="whitegrid")

# Data for plotting
metrics = {
    'Number of Playlists': num_playlists,
    'Number of Tracks': num_tracks,
    'Number of Unique Tracks': num_unique_tracks,
    'Number of Unique Albums': num_unique_albums,
    'Number of Unique Artists': num_unique_artists,
    'Number of Unique Playlist Titles': num_unique_playlist_titles,
    #'Number of Unique Normalized Playlist Titles': num_unique_normalized_playlist_titles,
    'Average Playlist Length (Number of tracks)': average_playlist_length
}

# Convert metrics to DataFrame for plotting
metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])

print(metrics_df)

#dataframe = dataframe.drop('name', axis=1)

#Distribution of number of tracks in playlists
plt.figure(figsize=(10, 6))
sns.histplot(playlists['num_tracks'], bins=50, kde=True)
plt.title('Distribution of Number of Tracks in Playlists')
plt.xlabel('Number of Tracks')
plt.ylabel('Frequency')
plt.savefig('figures/tracks_in_playlists.png')  

# Distribution of number of artists in playlists
plt.figure(figsize=(10, 6))
sns.histplot(playlists['num_artists'], bins=50, kde=True)
plt.title('Distribution of Number of Artists in Playlists')
plt.xlabel('Number of Artists')
plt.ylabel('Frequency')
plt.savefig('figures/artists_in_playlists.png')  

# Un box plot (o box-and-whisker plot) mostra la distribuzione dei dati
# quantitativi.

# plt.figure(figsize=(10, 6))
# sns.boxplot(data=dataframe, x = playlists['duration_ms'])
# plt.xlabel('Playlist duration (sec)')
# plt.savefig('figures/playlist_duration_boxplot.png')  

# mean = playlists['duration_ms'].mean()
# std = playlists['duration_ms'].std()
# lower_cut, upper_cut = mean -2 * std , mean +2 * std
# playlists = playlists[(playlists['duration_ms'] > lower_cut) & (playlists['duration_ms'] < upper_cut)]

# plt.figure(figsize=(10, 6))


# Convertire la durata da millisecondi a minuti
playlists['duration_min'] = playlists['duration_ms'] / (1000 * 60)

# Calcolo della media e dello standard deviation per filtrare gli outlier
mean = playlists['duration_min'].mean()
std = playlists['duration_min'].std()
lower_cut, upper_cut = mean - 2 * std, mean + 2 * std

# Filtrare le playlist che rientrano nel range desiderato (rimuovere outlier)
playlists_filtered = playlists[(playlists['duration_min'] > lower_cut) & (playlists['duration_min'] < upper_cut)]


# Istogramma per visualizzare la distribuzione della durata delle playlist filtrate
plt.figure(figsize=(10, 6))
sns.histplot(playlists_filtered['duration_min'], bins=50, kde=True, color='purple')
plt.xlabel('Playlist Duration (minutes)')
plt.ylabel('Frequency')
plt.title('Histogram of Playlist Duration (Filtered Data)')
plt.savefig('figures/playlist_duration_histogram_filtered.png')
plt.show()


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