from gensim.models import Word2Vec


w2v_model = Word2Vec.load('models/word2vec-trained-model.model')

input_playlist = ['75285', '2446', '7033', '29280', '1713']

input_embeddings = [w2v_model.wv[track_id] for track_id in input_playlist if track_id in w2v_model.wv]

recommended_tracks = []
for track_id in input_playlist:
    if track_id in w2v_model.wv:
        similar_tracks = w2v_model.wv.most_similar(track_id, topn=10)  # Trova 10 canzoni simili
        recommended_tracks.extend([track for track, similarity in similar_tracks])

# Elimina i duplicati
recommended_tracks = list(set(recommended_tracks))
print(f"Canali raccomandate: {recommended_tracks}")