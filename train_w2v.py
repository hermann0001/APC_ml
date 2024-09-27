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

#####################################
############ TRAINING ###############
#####################################

# Train Word2Vec model
model = Word2Vec(sentences=train_playlists, vector_size=300, window=5, min_count=1, workers=multiprocessing.cpu_count(), epochs=10, sg=0, negative=5, compute_loss=True)

loss_logger = LossLogger()
model.train(train_playlists, total_examples=model.corpus_count, epochs=model.epochs, callbacks=[loss_logger])

model.save(MDL_FOLDER + 'word2vec-trained-model.model')
print("Model trained and saved")