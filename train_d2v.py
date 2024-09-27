import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import multiprocessing
import logging
from utils import LossLogger

MDL_FOLDER = 'models/'
SRC_FOLDER = 'formatted/dataset/'

# load training set
train_playlists = pd.read_feather(SRC_FOLDER + 'train.feather')

# Set up logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Prepare training data (tag each playlist with its ID)
documents = []
for _, row in train_playlists.iterrows():
    playlist_id = str(row['playlist_id'])
    
    words=[f"{track}_{artist}_{pos}" for track, artist, pos in zip(row['track_id'], row['artist_id'], row['pos'])]

    documents.append(TaggedDocument(words=words, tags=[playlist_id]))


# Imposta i parametri per l'addestramento
# vector_size:  definisce la dimensionalità del vettore di output di parole. Ogni playlist (o word) sarà rappresentata da un vettore n-dimensionale.
#               Più sono grandi più sfumatore sono colte e maggiore è il carico computazionale
# window:       definisce quante words avanti e dietro vengono considerate dal modello per creare l'embedding, in pratica per ogni track vengono considerate le m tracce prima e dopo nella playlist
#               Grandi finestre possono catturare pattern più generali nelle playlist, piccole finestre si concentrano sulle relazioni strette tra tracce
# min_count:    Imposta la frequenza minima di una word per essere inclusa nel vocabolario
# workers:      Quanti processori lavorano parallelamente al training
# epoch:        Numero di iterazioni sull'intero corpus
# dm:           controlla l'algoritmo di training, per dm=1 usiamo Distributed Memory(DM), simile al modello Continuos Bag of Words (CBOW) usato per Word2Vec, per dm=0 si usa il Distributed Bag of Words(DBOW) simile al modello skip-gram per word2vec
model = Doc2Vec(vector_size=256, window=5, min_count=5, workers=multiprocessing.cpu_count(), epochs=10, dm=1, negative=5,compute_loss=True)

# Addestra il modello
model.build_vocab(documents)  # Costruisci il vocabolario

loss_logger = LossLogger()
model.train(documents, total_examples=model.corpus_count, epochs=model.epochs, callbacks=[loss_logger], compute_loss=True)

# Salva il modello
model.save(MDL_FOLDER + "d2v-trained-model.model")
print("Model trained and saved")