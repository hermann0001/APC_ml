import pandas as pd
import numpy as np
import gensim
from gensim.models import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity



df = pd.read_feather("/home/matteo/APC_ml/formatted/dataframe.feather")

unique_playlists = df.drop_duplicates(subset='playlist_id').head(1000)
print(unique_playlists.info())


documents = []
for _, row in unique_playlists.iterrows():
    documents.append(gensim.models.doc2vec.TaggedDocument(
        words=[str(row['playlist_name']), str(row['artist_name']), str(row['track_name'])],
        tags=[str(row['track_id'])]  # Usa track_id come tag per ogni documento
    ))

print(documents[:10])

# Imposta i parametri per l'addestramento
model = Doc2Vec(vector_size=400, window=2, min_count=1, workers=4, epochs=40)

# Addestra il modello
model.build_vocab(documents)  # Costruisci il vocabolario
model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)

# Salva il modello
model.save("d2v-trained-model.model")

print("end")