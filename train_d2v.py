import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import torch
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

#####################################
############ TRAINING ###############
#####################################

# check for gpu acceleration availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Initalizing training session using: {device}")

# Set up logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Prepare training data (tag each playlist with its ID)
documents = [TaggedDocument(words=playlist, tags=[str(pid)]) for pid, playlist in zip(train_playlists['playlist_id'], train_playlists['track_id'])]

# Imposta i parametri per l'addestramento
# vector_size:  definisce la dimensionalità del vettore di output di parole. Ogni playlist (o word) sarà rappresentata da un vettore n-dimensionale.
#               Più sono grandi più sfumatore sono colte e maggiore è il carico computazionale
# window:       definisce quante words avanti e dietro vengono considerate dal modello per creare l'embedding, in pratica per ogni track vengono considerate le m tracce prima e dopo nella playlist
#               Grandi finestre possono catturare pattern più generali nelle playlist, piccole finestre si concentrano sulle relazioni strette tra tracce
# min_count:    Imposta la frequenza minima di una word per essere inclusa nel vocabolario
# workers:      Quanti processori lavorano parallelamente al training
# epoch:        Numero di iterazioni sull'intero corpus
# dm:           controlla l'algoritmo di training, per dm=1 usiamo Distributed Memory(DM), simile al modello Continuos Bag of Words (CBOW) usato per Word2Vec, per dm=0 si usa il Distributed Bag of Words(DBOW) simile al modello skip-gram per word2vec
model = Doc2Vec(vector_size=300, window=5, min_count=1, workers=multiprocessing.cpu_count(), epochs=10, dm=1, compute_loss=True)

# Addestra il modello
model.build_vocab(documents)  # Costruisci il vocabolario
model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
print(f"Final training loss: {model.get_latest_training_loss()}")

# Salva il modello
model.save(MDL_FOLDER + "d2v-trained-model.model")

print("Model trained and saved")