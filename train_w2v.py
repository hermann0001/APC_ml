from gensim.models import Word2Vec
import pandas as pd
from sklearn.model_selection import train_test_split
import multiprocessing
import logging
from utils import Callback

MDL_FOLDER = 'models/'

train_playlists = []        # operazione di lettura da file train.boh

# prepare training data
train_data = [playlist for playlist in train_playlists['track_id']]

# Set up logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#####################################
############ TRAINING ###############
#####################################

# Train Word2Vec model
model = Word2Vec(sentences=train_data, vector_size=300, window=5, min_count=1, workers=multiprocessing.cpu_count(), epochs=10, sg=0)

callback = Callback()
model.train(train_playlists, total_examples=model.corpus_count, epochs=model.epochs, compute_loss=True, callbacks=[callback])

model.save(MDL_FOLDER + 'word2vec-trained-model.model')
print("Model trained and saved")