from gensim.models import Word2Vec
import pandas as pd
import multiprocessing
import logging
from utils import LossLogger
from datetime import datetime  # Importing datetime


MDL_FOLDER = 'models/'
SRC_FOLDER = 'formatted/dataset/'

train_playlists = pd.read_feather(SRC_FOLDER + 'train.feather')

# Set up logging
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # Format: YYYYMMDD_HHMMSS
log_filename = f'training_log_{timestamp}.txt'  # Log file name with timestamp
logger = logging.getLogger()
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(MDL_FOLDER + log_filename)
file_handler.setFormatter(logging.Formatter('%(asctime)s : %(levelname)s : %(message)s'))
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(asctime)s : %(levelname)s : %(message)s'))
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

train_data = []
for _, row in train_playlists.iterrows():
    words = [f"{track}_{artist}_{pos}" for track, artist, pos in zip(row['track_id'], row['artist_id'], row['pos'])]
    train_data.append(words)

logger.info("Loaded train data...")

#####################################
############ TRAINING ###############
#####################################

# Train Word2Vec model
model = Word2Vec(vector_size=300, window=20, min_count=1, workers=multiprocessing.cpu_count(), epochs=100, sg=0, negative=5)
model.build_vocab(train_data)

loss_logger = LossLogger()
model.train(train_data, total_examples=model.corpus_count, epochs=model.epochs,  compute_loss=True, callbacks=[loss_logger])

model.save(MDL_FOLDER + f'w2v/w2v-trained-model-{timestamp}.model')
logger.info("Model trained and saved")

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
