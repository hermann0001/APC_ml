from gensim.models import Word2Vec
from gensim.models import Doc2Vec
from collections import defaultdict


d2v_model = Doc2Vec.load('/home/matteo/APC_ml/models/d2v/d2v-trained-model.model')
w2v_model = Word2Vec.load('/home/matteo/APC_ml/models/w2v/w2v-trained-model.model')

input_playlist = ['75285', '2446', '7033', '29280', '1713']

input_embeddings = [w2v_model.wv[track_id] for track_id in input_playlist if track_id in w2v_model.wv]

if not input_embeddings:
    print("Nessuna delle canzoni dell'input è presente nel modello Word2Vec.")
else:
    input_playlist_vector = d2v_model.infer_vector(input_playlist)


input_playlist_vector = d2v_model.infer_vector(input_playlist)



recommended_tracks = defaultdict(float)

for track_id in input_playlist:
    if track_id in w2v_model.wv:
        similar_tracks = w2v_model.wv.most_similar(track_id, topn=10)  # Trova 10 canzoni simili
        for track, similarity in similar_tracks:
            recommended_tracks[track] += similarity  # Somma le similarità

# Ordina le canzoni raccomandate in base alla similarità aggregata
recommended_tracks = sorted(recommended_tracks.items(), key=lambda x: x[1], reverse=True)

# Mantieni solo le canzoni
recommended_tracks = [track for track, similarity in recommended_tracks]
print(f"Canali raccomandate: {recommended_tracks}")

# Trova playlist simili
similar_playlists = d2v_model.dv.most_similar([input_playlist_vector], topn=5)
print(f"Playlist simili: {similar_playlists}")

# Trova canzoni vicine basandoti sull'embedding della playlist
similar_tracks = w2v_model.wv.most_similar([input_playlist_vector], topn=10)
print(f"Canzoni raccomandate basate su Doc2Vec: {similar_tracks}")


#in questo modo sto trattando in modo separato le canzoni raccomandate da w2v e d2v 
#combined_recommendations = list(set(recommended_tracks + [track for track, sim in similar_tracks]))

#proviamo invece a combinare i punteggi di similaritá in modo ponderato, considerando le canzoni simili basate
#sia su w2v che su d2v e dando prioritá a quelle che appaiono in entrambi i risultati
combined_recommendations = defaultdict(float)

for track, sim in recommended_tracks:
    combined_recommendations[track] += sim

for track, sim in similar_tracks:
    combined_recommendations[track] += sim

combined_recommendations = sorted(combined_recommendations.items(), key=lambda x: x[1], reverse=True)
combined_recommendations = [track for track, similarity in combined_recommendations]

print(combined_recommendations)