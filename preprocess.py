import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

SAMPLE_SIZE = 100000 # no. of rows loaded (total = 1M)
PLAYLIST_CSV = "temp/playlists.csv"
TRACK_CSV = "temp/tracks.csv"

def normalize_name(name):
    name = name.lower()
    name = re.sub(r"[.,#!$%\^\*;:{}=\_`~()@]", ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def extract_uri(uri):
    if isinstance(uri, str) and ':' in uri:
        parts = uri.split(':')
        return parts[2] if len(parts) > 2 else None
    return None

playlists_df = pd.read_csv(PLAYLIST_CSV)#, nrows=SAMPLE_SIZE)
tracks_df = pd.read_csv(TRACK_CSV)#, nrows=SAMPLE_SIZE)

playlists_df = playlists_df.rename(columns={'duration_ms':'playlist_duration_ms'})
tracks_df = tracks_df.rename(columns={'duration_ms':'track_duration_ms'})

dataframe = tracks_df.merge(playlists_df, on='pid', how='left')


print("Original DataFrame")
print(dataframe.memory_usage().sum())

# Convert into smaller data types
dataframe['pid'] = pd.to_numeric(dataframe['pid'], downcast='integer')
dataframe['num_tracks'] = pd.to_numeric(dataframe['num_tracks'], downcast='integer')
dataframe['num_albums'] = pd.to_numeric(dataframe['num_albums'], downcast='integer')
dataframe['num_followers'] = pd.to_numeric(dataframe['num_followers'], downcast='integer')
dataframe['num_edits'] = pd.to_numeric(dataframe['num_edits'], downcast='integer')
dataframe['modified_at'] = pd.to_datetime(dataframe['modified_at'], unit='s')
dataframe['num_artists'] = pd.to_numeric(dataframe['num_artists'], downcast='integer')
dataframe['track_uri'] = dataframe['track_uri'].apply(extract_uri)
dataframe['artist_uri'] = dataframe['artist_uri'].apply(extract_uri)
dataframe['album_uri'] = dataframe['album_uri'].apply(extract_uri)
dataframe['normalized_playlist_title'] = dataframe['name'].apply(normalize_name)


print("Dropped DataFrame")
print(dataframe.memory_usage().sum())


######################################################

# Number of playlists
num_playlists = dataframe['pid'].nunique()

# Number of tracks
num_tracks = dataframe['track_uri'].nunique()

# Number of unique tracks
num_unique_tracks = dataframe['track_uri'].nunique()

# Number of unique albums
num_unique_albums = dataframe['album_uri'].nunique()

# Number of unique artists
num_unique_artists = dataframe['artist_uri'].nunique()

# Number of unique playlist titles
num_unique_playlist_titles = dataframe['name'].nunique()

# Number of unique normalized playlist titles
num_unique_normalized_playlist_titles = dataframe['normalized_playlist_title'].nunique()

# Average playlist length (number of tracks)
average_playlist_length = dataframe['num_tracks'].mean()

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
    'Number of Unique Normalized Playlist Titles': num_unique_normalized_playlist_titles,
    'Average Playlist Length (Number of tracks)': average_playlist_length
}

# Convert metrics to DataFrame for plotting
metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])

print(metrics_df)

# Distribution of number of tracks in playlists
plt.figure(figsize=(10, 6))
sns.histplot(dataframe['num_tracks'], bins=50, kde=True)
plt.title('Distribution of Number of Tracks in Playlists')
plt.xlabel('Number of Tracks')
plt.ylabel('Frequency')
plt.savefig('./fig1.png')  # Adjust path as needed

# Distribution of number of artists in playlists
plt.figure(figsize=(10, 6))
sns.histplot(dataframe['num_artists'], bins=50, kde=True)
plt.title('Distribution of Number of Artists in Playlists')
plt.xlabel('Number of Artists')
plt.ylabel('Frequency')
plt.savefig('./fig2.png')  # Adjust path as needed


######### Split Dataset
