import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import Word2Vec
import logging

MDL_FOLDER = 'models/'
SRC_FOLDER = 'formatted/dataset/'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

model = Word2Vec.load(MDL_FOLDER + 'w2v/w2v-trained-model.model')

embedding_matrix = model.wv[model.wv.index_to_key]
logging.info(f'Embedding matrix shape: {embedding_matrix.shape}')


def spherical_kmeans(X, n_clusters, max_iter=300):
    logging.info(f'Starting Spherical K-Means with {n_clusters} clusters.')
    
    # Normalize the data
    X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)
    
    # Initialize KMeans with spherical data
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=max_iter)
    kmeans.fit(X_normalized)
    
    logging.info(f'Finished Spherical K-Means with {n_clusters} clusters.')
    return kmeans.labels_

def locate_optimal_elbow(x, y):
    logging.info('Calculating optimal elbow point.')
    
    if x.empty or y.empty:
        raise ValueError("Input Series cannot be empty")

    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    # Connect the first and last point of the curve with a straight line
    line_start = (x[0], y[0])
    line_end = (x.iloc[-1], y.iloc[-1])
    
    # Calculate distances from points to the line
    distances = []
    for i in range(len(y)):
        numerator = abs((line_end[1] - line_start[1]) * x[i] - (line_end[0] - line_start[0]) * y[i] + 
                        line_end[0] * line_start[1] - line_end[1] * line_start[0])
        denominator = np.sqrt((line_end[1] - line_start[1]) ** 2 + (line_end[0] - line_start[0]) ** 2)
        distances.append(numerator / denominator if denominator != 0 else 0)

    # The index of the maximum distance corresponds to the elbow point
    elbow_index = np.argmax(distances)
    logging.info(f'Optimal elbow point found at index: {elbow_index}')
    return elbow_index


wcss = []  # List to hold Within-Cluster Sum of Squares (WCSS)
for n_clusters in range(10, 51, 10):  # Iterate over number of clusters
    logging.info(f'Calculating WCSS for {n_clusters} clusters.')
    labels = spherical_kmeans(embedding_matrix, n_clusters)
    # Calculate WCSS
    wcss.append(sum((np.linalg.norm(embedding_matrix[labels == i] - np.mean(embedding_matrix[labels == i], axis=0)))**2 for i in range(n_clusters)))

# Create DataFrame to analyze WCSS
skm_df = pd.DataFrame({'WCSS': wcss, 'n_clusters': range(10, 151, 10)})

# Locate optimal elbow
k_opt = locate_optimal_elbow(skm_df['n_clusters'], skm_df['WCSS'])
skm_opt_labels = spherical_kmeans(embedding_matrix, k_opt)

# After calculating WCSS
plt.figure(figsize=(10, 6))
plt.plot(range(10, 201, 10), wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.axvline(x=k_opt + 10, linestyle='--', color='red', label='Optimal k')
plt.legend()
plt.savefig('figures/elbow_method.png')

# Prepare the DataFrame for songs with their clusters
songs_cluster = pd.DataFrame(index=model.wv.index_to_key, columns=['cluster'])
songs_cluster['cluster'] = skm_opt_labels
songs_cluster['cluster'] = songs_cluster['cluster'].fillna(-1).astype(int).astype('category')

# Visualization using t-SNE
logging.info('Performing t-SNE visualization...')
embedding_tsne_full = TSNE(n_components=2, metric='cosine', random_state=123).fit_transform(embedding_matrix)

# Prepare DataFrame for plotting
tsne_df_full = pd.DataFrame(embedding_tsne_full, columns=['x', 'y'])
tsne_df_full['cluster'] = songs_cluster['cluster'].values

# Plotting
plt.figure(figsize=(12, 8))
sns.scatterplot(data=tsne_df_full, x='x', y='y', hue='cluster', palette='viridis', legend='full', alpha=0.7)
plt.title('t-SNE Visualization of All Song Clusters')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig('figures/full-tsne.png')

logging.info('Full t-SNE visualization completed.')

# Step 2: Perform t-SNE on a random subset of clusters
logging.info('Performing t-SNE visualization on a random subset of clusters...')

# Randomly select 10 unique clusters
unique_clusters = songs_cluster['cluster'].cat.categories
selected_clusters = np.random.choice(unique_clusters[unique_clusters != -1], size=10, replace=False)

# Filter embeddings for the selected clusters
filtered_indices = songs_cluster[songs_cluster['cluster'].isin(selected_clusters)].index
filtered_embeddings = embedding_matrix[[model.wv.key_to_index[key] for key in filtered_indices]]

# Perform t-SNE on the filtered embeddings
embedding_tsne_filtered = TSNE(n_components=2, metric='cosine', random_state=123).fit_transform(filtered_embeddings)

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
plt.savefig('figures/selected-tsne.png')

logging.info('t-SNE visualization for selected clusters completed.')

logging.info(f"The optimal number of clusters (elbow point) is: {k_opt + 10}")  # Adjust index since we started from 10
