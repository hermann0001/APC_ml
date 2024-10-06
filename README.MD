# Automatic Music Playlist Continuator

Codice python per addestrare un modello di Machine Learning sul dataset reso disponibile dalla RecSys Challenge 2018 contenente un milione di playlist. \
Info: [C.W. Chen, P. Lamere, M. Schedl, and H. Zamani. Recsys Challenge 2018: Automatic Music Playlist Continuation. In Proceedings of the 12th ACM Conference on Recommender Systems (RecSys ’18), 2018.](https://dl.acm.org/doi/abs/10.1145/3240323.3240342)

# Requisiti

* Il codice è stato scritto e compilato con Python 3.10.12
* I pacchetti e librerie necessari da installare sono scritti nel file _requirements.txt_

```bash
pip3 install -r requirements.txt
```

# Uso

## Setup dell'ambiente

* Esegui `setup.sh` per creare la struttura delle directory necessaria.

```text
📦APC_ML
 ┣ 📂figures
 ┣ 📂formatted
 ┃ ┗ 📂dataset
 ┣ 📂models
 ┃ ┗ 📂w2v
 ┣ 📜MetaSpotifyDataExtractor.py
 ┣ 📜predict_w2v.ipynb
 ┣ 📜predict_w2v.py
 ┣ 📜preprocess_dataset.py
 ┣ 📜read_dataset.py
 ┣ 📜requirements.txt
 ┣ 📜song2vec.ipynb
 ┣ 📜stats.ipynb
 ┣ 📜train_w2v.ipynb
 ┗ 📜utils.py
```

* **dataset** è la cartella destinata a contenere il dataset estratto direttamente dentro di essa.
* **figures** contiene vari grafici o qualsiasi file `.png` generato dal codice.
* **formatted** contiene i vari dataset convertiti e allegeriti in file `.csv` generati e preprocessati dal codice python.
* **models** conteine i modelli addestrati da Word2Vec

### Struttura dei dati

* `artists.csv`

    ```text
    artist_id | artist_uri | artist_name | 
    ```

* `playlists.csv`

    ```text
    playlist_id | name | num_tracks | num_artists | num_albums | num_followers | num_edits | duration_ms | modified_at | collaborative | description|
    ```

* `tracks.csv`

    ```text
    track_id | track_uri | track_name | artist_id | album_uri | duration_ms_track | album_name |
    ```

* `playlist_tracks.csv`

    ```text
    playlist_id | track_id | artist_id | pos |
    ```
