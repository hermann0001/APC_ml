from gensim.models import Word2Vec
import pandas as pd
from sklearn.model_selection import train_test_split
import multiprocessing
import logging

MDL_FOLDER = 'models/'


#####################################
########### PREPARATION #############
#####################################

# load preprocessed dataset
df = pd.read_feather("formatted/dataframe.feather")

# group tracks by playlists to form documents
playlist_documents = df.groupby('playlist_id')['track_id'].apply(list).reset_index()

# split train and test set
train_playlists, test_playlists = train_test_split(playlist_documents, test_size=0.2, random_state=666)
print(f"Train playlists: {len(train_playlists)}, Test Playlists: {len(test_playlists)}")

# prepare training data
train_data = [playlist for playlist in train_playlists['track_id']]

# Set up logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#####################################
############ TRAINING ###############
#####################################

# Train Word2Vec model
model = Word2Vec(sentences=train_data, vector_size=300, window=5, min_count=1, workers=multiprocessing.cpu_count(), epochs=10, dm=1, compute_loss=True)
print(f"Final training loss: {model.get_latest_training_loss()}")


model.save(MDL_FOLDER + 'word2vec-trained-model.model')
print("Model trained and saved")