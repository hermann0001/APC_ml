import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

SRC_FOLDER = 'formatted/dataset/'
FIGURE_FOLDER = 'figures/'

def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    array = np.array(array)  # Convert to NumPy array
    if np.amin(array) < 0:
        array -= np.amin(array)  # Make all values non-negative
    array = array.astype(float)  # Ensure float division
    array += 1e-7  # Prevent division by zero

    array = np.sort(array)  # Sort the array
    n = len(array)
    cumulative_sum = np.cumsum(array)  # Cumulative sum of the sorted array
    gini_coefficient = (n + 1 - 2 * np.sum(cumulative_sum) / cumulative_sum[-1]) / n
    return gini_coefficient

def artist_heterogeneity(playlist_tracks):
    unique_artists = len(set(playlist_tracks['artist_id']))
    unique_tracks = len(set(playlist_tracks['track_id']))

    if unique_artists > 0:
        return np.log2(unique_tracks/unique_artists)
    else:
        return np.nan
def compute_metrics(df_playlists_tracks):
    playlist_metrics = []
    results = {'playlist_id': [], 'sample_type': [], 'artist_heterogeneity': []}

    for playlist_id, group in df_playlists_tracks.groupby('playlist_id'):
        artist_ids = group['artist_id'].tolist()
        track_ids = group['track_id'].tolist()
        unique_artists = len(set(artist_ids))
        unique_tracks = len(set(track_ids))
        total_tracks = len(artist_ids)
        
        # Artist repeat rate (consecutive repeats)
        repeat_count = sum(artist_ids[i] == artist_ids[i + 1] for i in range(len(artist_ids) - 1))
        repeat_rate = repeat_count / (total_tracks - 1) if total_tracks > 1 else 0

        # Artist concentration index (max count of any artist / total tracks)
        artist_counts = Counter(artist_ids)
        concentration_index = max(artist_counts.values()) / total_tracks
        
        # Gini coefficient for artist distribution
        artist_freq = list(artist_counts.values())
        artist_gini = gini(artist_freq)

        # Jaccard similarity between first half and second half
        half_index = total_tracks // 2
        first_half_artists = set(artist_ids[:half_index])
        second_half_artists = set(artist_ids[half_index:])
        jaccard_similarity = len(first_half_artists & second_half_artists) / len(first_half_artists | second_half_artists) if len(first_half_artists | second_half_artists) > 0 else 0

        # Calcolo dell'eterogeneità artistica
        heterogeneity_order = artist_heterogeneity(group.head(30))
        if len(group) > 30:
            random_tracks = group.sample(30)
        else:
            random_tracks = group
        heterogeneity_random = artist_heterogeneity(random_tracks)
        heterogeneity_total = artist_heterogeneity(group)

        # Aggiungi dati al dizionario results per ogni tipo di campione
        results['playlist_id'].append(playlist_id)
        results['sample_type'].append('First 30 Tracks')
        results['artist_heterogeneity'].append(heterogeneity_order)        

        results['playlist_id'].append(playlist_id)
        results['sample_type'].append('Random 30 Tracks')
        results['artist_heterogeneity'].append(heterogeneity_random)

        results['playlist_id'].append(playlist_id)
        results['sample_type'].append('Entire Playlist')
        results['artist_heterogeneity'].append(heterogeneity_total)

        # Store metrics
        playlist_metrics.append({
            'playlist_id': playlist_id,
            'unique_artists': unique_artists,
            'unique_tracks': unique_tracks,
            'repeat_rate': repeat_rate,
            'concentration_index': concentration_index,
            'gini_coefficient': artist_gini,
            'jaccard_similarity': jaccard_similarity,
        })

    # Dopo aver raccolto tutte le metriche, trasformiamo `results` in un DataFrame
    heterogeneity_df = pd.DataFrame(results)

    return playlist_metrics, heterogeneity_df

def plot_metrics(df_playlist_metrics, arH):
    # Set up the plotting style
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(18, 12))

    # 1. Histogram of Unique Artists per Playlist
    plt.subplot(3, 2, 1)
    sns.histplot(df_playlist_metrics['unique_artists'], bins=30, kde=True, color='steelblue', edgecolor='black', alpha=0.7)
    plt.title('Distribution of Unique Artists per Playlist')
    plt.xlabel('Unique Artists')
    plt.ylabel('Frequency')

    # 2. Histogram of Unique Tracks per Playlist
    plt.subplot(3, 2, 2)
    sns.histplot(df_playlist_metrics['unique_tracks'], bins=30, kde=True, color='mediumseagreen', edgecolor='black', alpha=0.7)
    plt.title('Distribution of Unique Tracks per Playlist')
    plt.xlabel('Unique Tracks')
    plt.ylabel('Frequency')

    # 3. Histogram of Gini Coefficients
    plt.subplot(3, 2, 3)
    sns.histplot(df_playlist_metrics['gini_coefficient'], bins=30, kde=True, color='darkorange', edgecolor='black', alpha=0.7)
    plt.title('Distribution of Gini Coefficients')
    plt.xlabel('Gini Coefficient')
    plt.ylabel('Frequency')

    # 4. Box Plot: Repeat Rate Distribution
    plt.subplot(3, 2, 4)
    sns.boxplot(x=df_playlist_metrics['repeat_rate'], color='lightcoral', linewidth=2)
    plt.title('Repeat Rate Distribution')
    plt.xlabel('Repeat Rate')

    # 5. Scatter Plot: Concentration Index vs Gini Coefficient
    plt.subplot(3, 2, 5)
    sns.scatterplot(x='concentration_index', y='gini_coefficient', data=df_playlist_metrics, s=100, color='royalblue', alpha=0.7)
    plt.title('Concentration Index vs Gini Coefficient')
    plt.xlabel('Concentration Index')
    plt.ylabel('Gini Coefficient')

    # 6. Histogram of Jaccard Similarities
    plt.subplot(3, 2, 6)
    sns.histplot(df_playlist_metrics['jaccard_similarity'], bins=30, kde=True, color='tomato', edgecolor='black', alpha=0.7)
    plt.title('Distribution of Jaccard Similarities')
    plt.xlabel('Jaccard Similarity')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(FIGURE_FOLDER + 'metrics.png')

    # Usa il DataFrame arH per il boxplot dell'eterogeneità artistica
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='sample_type', y='artist_heterogeneity', data=arH, hue='sample_type', palette='coolwarm', linewidth=2.5, fliersize=3, dodge=False)
    plt.title('Artist Heterogeneity by Sampling Method')
    plt.xlabel('Sampling Method')
    plt.ylabel('Artist Heterogeneity')
    plt.savefig(FIGURE_FOLDER + 'artist_heterogeneity.png')


def reduce_size(dataframe):
    print("Original DataFrame")
    print(dataframe.memory_usage().sum())

    for col in dataframe.select_dtypes(include=[np.number]).columns:
        if pd.api.types.is_integer_dtype(dataframe[col]):
            dataframe[col] = pd.to_numeric(dataframe[col], downcast='integer')
        elif pd.api.types.is_float_dtype(dataframe[col]):
            dataframe[col] = pd.to_numeric(dataframe[col], downcast='float')

    print("Reduced DataFrame")
    print(dataframe.memory_usage().sum())
    return dataframe


if __name__ == '__main__':
    df_artists = pd.read_csv(SRC_FOLDER + 'artists.csv')#, nrows=SAMPLE)
    df_tracks =  pd.read_csv(SRC_FOLDER + 'tracks.csv')#, nrows=SAMPLE)
    df_playlists =  pd.read_csv(SRC_FOLDER + 'playlists.csv')#, nrows=SAMPLE)
    df_playlists_tracks = pd.read_csv(SRC_FOLDER + 'playlist_tracks.csv')#, nrows=SAMPLE)

    print(df_artists.head())
    print(df_artists.info())
    print("\n\n")
    print(df_tracks.head())
    print(df_tracks.info())
    print("\n\n")
    print(df_playlists.head())
    print(df_playlists.info())
    print("\n\n")
    print(df_playlists_tracks.head())
    print(df_playlists_tracks.info())

    df_artists = reduce_size(df_artists)
    df_tracks = reduce_size(df_tracks)
    df_playlists = reduce_size(df_playlists)
    df_playlists_tracks = reduce_size(df_playlists_tracks)

    playlist_metrics, arH = compute_metrics(df_playlists_tracks)
    df_playlist_metrics = pd.DataFrame(playlist_metrics)
    
    plot_metrics(df_playlist_metrics, arH)

