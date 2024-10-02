import numpy as np
import pandas as pd
from cuml.cluster import KMeans as cuKMeans
from cuml.manifold import TSNE as cuTSNE
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import Word2Vec
from datetime import datetime
from sklearn.preprocessing import normalize
import logging
import math
import tqdm
import random

MDL_FOLDER = 'models/'
SRC_FOLDER = 'formatted/dataset/'
model_timestamp = '20241002_021349'

def locate_optimal_elbow(x, y):
    # START AND FINAL POINTS
    p1 = (x[0], y[0])
    p2 = (x[-1], y[-1])
    
    # EQUATION OF LINE: y = mx + c
    m = (p2[1] - p1[1]) / (p2[0] - p1[0])
    c = (p2[1] - (m * p2[0]))
    
    # DISTANCE FROM EACH POINTS TO LINE mx - y + c = 0
    a, b = m, -1
    dist = np.array([abs(a*x0+b*y0+c)/math.sqrt(a**2+b**2) for x0, y0 in zip(x,y)])
    return x[np.argmax(dist)]


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load Word2Vec model
model = Word2Vec.load(MDL_FOLDER + f'w2v/w2v-trained-model-{model_timestamp}.model')

# Get the embedding matrix
embedding_matrix = model.wv[model.wv.index_to_key]
logging.info(f'Embedding matrix shape: {embedding_matrix.shape}')


range_k_clusters = (10, 500)
km_list = []
for k in tqdm(range(*range_k_clusters, 10)):
    normalized_embedding_matrix = normalize(embedding_matrix)

    km = cuKMeans(n_clusters = k, n_init = 5, random_state = 666).fit(normalized_embedding_matrix)
    
    result_dict = {
        "k": k,
        "WCSS": km.inertia_,
        "km_object": km
    }
    
    km_list.append(result_dict)
km_df = pd.DataFrame(km_list).set_index('k')

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
km_df.WCSS.plot()
plt.xlabel("No. of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method", fontweight = "bold")
plt.savefig(f'figures/elbow_method_{timestamp}.png')

# Locate optimal elbow
k_opt = locate_optimal_elbow(km_df.index, km_df['WCSS'].values)

#k_opt = 100
km_opt_labels, _, _ = km_df.loc[k_opt, 'km_object']

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
km_df.WCSS.plot()
plt.xlabel("No. of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method", fontweight = "bold")
plt.axvline(x=k_opt, linestyle='--', color='red', label='Optimal k')
plt.legend()
plt.savefig(f'figures/elbow_method_{timestamp}.png')

songs = pd.read_feather(SRC_FOLDER + 'dataframe.feather')
songs.drop_duplicates(subset=['track_id'], inplace=True)
print(songs.head())

songs.loc[model.wv.index_to_key, 'cluster'] = km_opt_labels
songs['cluster'] = songs['cluster'].fillna(-1).astype(int).astype('category')

# Visualization using t-SNE
logging.info('Performing t-SNE visualization...')
embedding_tsne_full = cuTSNE(n_components=2, metric='cosine', random_state=666).fit_transform(embedding_matrix)

# Prepare DataFrame for plotting
songs.loc[model.wv.index_to_key, 'x'] = embedding_tsne_full[:,0]
songs.loc[model.wv.index_to_key, '1'] = embedding_tsne_full[:,1]

# Plotting full t-SNE visualization
plt.figure(figsize=(12, 8))
sns.scatterplot(data=songs[songs['cluster'] != -1], x='x', y='y', hue='cluster', palette='viridis', legend=False)
plt.title(f't-SNE Visualization of {k_opt} Song Clusters', fontweight='bold')
plt.savefig(f'figures/full-tsne_{timestamp}.png')

logging.info('Full t-SNE visualization completed.')

# Step 2: Perform t-SNE on a random subset of clusters
logging.info('Performing t-SNE visualization on a random subset of clusters...')

# Randomly select 10 unique clusters
random.seed(100)
random_cluster2plot = random.sample(range(k_opt), 10)
random_songs = songs[songs.cluster.isin(random_cluster2plot)].copy()
random_tsne = cuTSNE(n_components = 2, metric = 'cosine', random_state = 100).fit_transform(model.wv[random_songs.index])
random_songs.loc[random_songs.index, 'x'] = random_tsne[:,0]
random_songs.loc[random_songs.index, 'y'] = random_tsne[:,1]

plt.figure(figsize=(12, 8))
g = sns.scatterplot(data = random_songs,
                x = 'x', y = 'y', palette = "viridis",
                hue = 'cluster')
g.legend(loc = "upper left", bbox_to_anchor = (1, 1))
g.set_title(f"Randomly selected {len(random_cluster2plot)} Song Clusters", fontweight = "bold")
plt.savefig(f'figures/selected-tsne_{timestamp}.png')
