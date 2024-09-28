from gensim.models import Word2Vec
import pandas as pd
import multiprocessing
import logging
from utils import LossLogger

MDL_FOLDER = 'models/'
SRC_FOLDER = 'formatted/dataset/'

train_playlists = pd.read_feather(SRC_FOLDER + 'train.feather')

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
model = Word2Vec(vector_size=400, window=10, min_count=3, workers=multiprocessing.cpu_count(), epochs=150, sg=0, negative=5, alpha=0.1)
model.build_vocab(train_data)

loss_logger = LossLogger()
model.train(train_data, total_examples=model.corpus_count, epochs=model.epochs,  compute_loss=True, callbacks=[loss_logger])

print(loss_logger.losses)

model.save(MDL_FOLDER + 'w2v-trained-model.model')
print("Model trained and saved")

# Epoch 0 Loss: 16777216.0
# Epoch 1 Loss:     1032.0
# Epoch 2 Loss:    43708.0
# Epoch 3 Loss:  1075678.0
# Epoch 4 Loss:  4293980.0
# Epoch 5 Loss:  5617342.0
# Epoch 6 Loss:  5768688.0
# Epoch 7 Loss:   791208.0
# Epoch 8 Loss:   748644.0
# Epoch 9 Loss:   709400.0