import numpy as np
import pandas as pd
from cuml.cluster import KMeans as cuKMeans
from cuml.manifold import TSNE as cuTSNE
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import Word2Vec
import logging
from datetime import datetime

MDL_FOLDER = 'models/'
SRC_FOLDER = 'formatted/dataset/'

model_timestamp = '20241001_182949'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load Word2Vec model
model = Word2Vec.load(MDL_FOLDER + f'w2v/w2v-trained-model_{model_timestamp}.model')

# Get the embedding matrix
embedding_matrix = model.wv[model.wv.index_to_key]
logging.info(f'Embedding matrix shape: {embedding_matrix.shape}')


def spherical_kmeans(X, n_clusters, max_iter=300):
    logging.info(f'Starting Spherical K-Means with {n_clusters} clusters.')
    
    # Normalize the data
    X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)
    
    # Initialize KMeans with spherical data
    kmeans = cuKMeans(n_clusters=n_clusters, init='k-means++', max_iter=max_iter)
    kmeans.fit(X_normalized)
    
    logging.info(f'Finished Spherical K-Means with {n_clusters} clusters.')
    return kmeans.labels_, kmeans.cluster_centers_, kmeans.inertia_


def locate_optimal_elbow(x, y):
    logging.info('Calculating optimal elbow point.')
    
    if x.empty or y.empty:
        raise ValueError("Input Series cannot be empty")

    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    # Normalize x and y for scale invariance
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())

    # Coordinates of the first and last points
    p1 = np.array([x_norm.iloc[0], y_norm.iloc[0]])
    p2 = np.array([x_norm.iloc[-1], y_norm.iloc[-1]])

    # Compute distances from each point to the line p1 -> p2
    distances = []
    for i in range(len(x_norm)):
        p = np.array([x_norm.iloc[i], y_norm.iloc[i]])
        distance = np.linalg.norm(np.cross(p2 - p1, p1 - p)) / np.linalg.norm(p2 - p1)
        distances.append(distance)

    # Find the index of the point with the maximum distance
    elbow_index = np.argmax(distances)

    # Log the elbow point found
    logging.info(f'Optimal elbow point found at index: {elbow_index}')
    
    return elbow_index

# Calculate Within-Cluster Sum of Squares (WCSS)
wcss = []  # List to hold Within-Cluster Sum of Squares (WCSS)
for n_clusters in range(10, 401, 10):  # Iterate over number of clusters
    logging.info(f'Calculating WCSS for {n_clusters} clusters.')
    labels, centers, inertia = spherical_kmeans(embedding_matrix, n_clusters)
    wcss.append(inertia)  # Inertia is equivalent to WCSS

# Create DataFrame to analyze WCSS
skm_df = pd.DataFrame({'WCSS': wcss, 'n_clusters': range(10, 401, 10)})

# Locate optimal elbow
k_opt = locate_optimal_elbow(skm_df['n_clusters'], skm_df['WCSS'])
skm_opt_labels, _, _ = spherical_kmeans(embedding_matrix, k_opt)

# Plot Elbow Method
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.figure(figsize=(10, 6))
plt.plot(range(10, 401, 10), wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.axvline(x=k_opt + 10, linestyle='--', color='red', label='Optimal k')
plt.legend()
plt.savefig(f'figures/elbow_method_{timestamp}.png')

# Prepare the DataFrame for songs with their clusters
songs_cluster = pd.DataFrame(index=model.wv.index_to_key, columns=['cluster'])
songs_cluster['cluster'] = skm_opt_labels
songs_cluster['cluster'] = songs_cluster['cluster'].fillna(-1).astype(int).astype('category')

# Visualization using t-SNE
logging.info('Performing t-SNE visualization...')
embedding_tsne_full = cuTSNE(n_components=2, perplexity=30, n_iter=1000, metric='cosine', random_state=123).fit_transform(embedding_matrix)

# Prepare DataFrame for plotting
tsne_df_full = pd.DataFrame(embedding_tsne_full, columns=['x', 'y'])
tsne_df_full['cluster'] = songs_cluster['cluster'].values

# Plotting full t-SNE visualization
plt.figure(figsize=(12, 8))
sns.scatterplot(data=tsne_df_full, x='x', y='y', hue='cluster', palette='viridis', legend='full', alpha=0.7)
plt.title('t-SNE Visualization of All Song Clusters')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig(f'figures/full-tsne_{timestamp}.png')

logging.info('Full t-SNE visualization completed.')

# Step 2: Perform t-SNE on a random subset of clusters
logging.info('Performing t-SNE visualization on a random subset of clusters...')

# Randomly select 10 unique clusters
unique_clusters = songs_cluster['cluster'].cat.categories
selected_clusters = np.random.choice(unique_clusters[unique_clusters != -1], size=5, replace=False)

# Filter embeddings for the selected clusters
filtered_indices = songs_cluster[songs_cluster['cluster'].isin(selected_clusters)].index
filtered_embeddings = embedding_matrix[[model.wv.key_to_index[key] for key in filtered_indices]]

# Perform t-SNE on the filtered embeddings
embedding_tsne_filtered = cuTSNE(n_components=2, perplexity=10, n_iter=1000, metric='cosine', random_state=123).fit_transform(filtered_embeddings)

# Prepare DataFrame for plotting
tsne_df_filtered = pd.DataFrame(embedding_tsne_filtered, columns=['x', 'y'])
tsne_df_filtered['cluster'] = songs_cluster.loc[filtered_indices, 'cluster'].values

# Plotting
plt.figure(figsize=(12, 8))
sns.scatterplot(data=tsne_df_filtered, x='x', y='y', hue='cluster', palette='viridis', legend='full', alpha=0.7)
plt.title('t-SNE Visualization of Selected Song Clusters')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig(f'figures/selected-tsne_{timestamp}.png')

# Zoomed-in plot for specific t-SNE ranges
subset = tsne_df_filtered[(tsne_df_filtered['x'] > -1000) & (tsne_df_filtered['x'] < 1000)]

plt.scatter(subset['x'], subset['y'], c=subset['cluster'].cat.codes, cmap='tab10')
plt.title("Zoomed-in t-SNE Selected Visualization")
plt.show()

logging.info('t-SNE visualization for selected clusters completed.')
