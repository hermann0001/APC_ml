from gensim.models import Word2Vec
import pandas as pd
import multiprocessing
import logging
from utils import LossLogger
from datetime import datetime  # Importing datetime
import pickle
import matplotlib.pyplot as plt

MDL_FOLDER = 'models/'
SRC_FOLDER = 'formatted/dataset/'

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

with open(SRC_FOLDER + 'train.pkl', 'rb') as file:
    train_data = pickle.load(file)

print(len(train_data))
# Train Word2Vec model
model = Word2Vec(vector_size=300, window=5, min_count=5, workers=multiprocessing.cpu_count(), epochs=500, sg=0, negative=5, alpha=0.05)
model.build_vocab(train_data)

loss_logger = LossLogger()
model.train(train_data, total_examples=model.corpus_count, epochs=model.epochs,  compute_loss=True, callbacks=[loss_logger])

model.save(MDL_FOLDER + f'w2v/w2v-trained-model-{timestamp}.model')
logger.info("Model trained and saved")

#questo po essere sostituito dalla lettura di un file
log_data = loss_logger.losses

epochs = list(log_data.keys())
losses = list(log_data.values())

# Creazione del grafico
log_data = loss_logger.losses

epochs = list(log_data.keys())
losses = list(log_data.values())

# Creazione del grafico
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(epochs, losses, marker='o', linestyle='-', color='b')
fig.suptitle(rf'Loss function per $\alpha = {model.alpha}$', fontsize=14, fontweight='bold') # Titolo con la lettera greca alpha
ax.grid(True)
ax.set_xticks(range(0, model.epochs, 25)) 
ax.set_xlabel('Epochs')  # Etichetta asse X
ax.set_ylabel('Loss')    # Etichetta asse Y
plt.savefig(f'figures/w2v_loss_plot-{timestamp}.png')