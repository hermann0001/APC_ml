from gensim.models import Word2Vec
import pandas as pd
import multiprocessing
import logging
from utils import LossLogger

MDL_FOLDER = 'models/'
SRC_FOLDER = 'formatted/dataset/'

train_playlists = pd.read_feather(SRC_FOLDER + 'train.feather')

print(train_playlists)


# Set up logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

train_data = []
for _, row in train_playlists.iterrows():
    words = [f"{track}_{artist}_{pos}" for track, artist, pos in zip(row['track_id'], row['artist_id'], row['pos'])]
    train_data.append(words)

#####################################
############ TRAINING ###############
#####################################

# Train Word2Vec model
model = Word2Vec(sentences=train_data, vector_size=300, window=5, min_count=5, workers=multiprocessing.cpu_count(), epochs=10, sg=0, negative=5)
model.build_vocab(train_playlists)

loss_logger = LossLogger()
model.train(train_data, total_examples=model.corpus_count, epochs=model.epochs,  compute_loss=True, callbacks=[loss_logger])

model.save(MDL_FOLDER + 'w2v-trained-model.model')
print("Model trained and saved")