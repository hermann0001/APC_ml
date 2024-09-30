import pandas as pd
import numpy as np
import argparse
import logging
from gensim.models import Doc2Vec, Word2Vec
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import LabelBinarizer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
MDL_FOLDER = 'models/'
SRC_FOLDER = 'formatted/dataset/'


def mean_vectors(tracks, model):
    vec = []
    for track_id in tracks:
        try:
            vec.append(model.wv(track_id))
        except KeyError:
            continue
    return np.mean(vec, axis=0) if vec else None


def get_similar_tracks(model, track_id, top_n=10):
    if isinstance(model, Word2Vec):
        if track_id not in model.wv:
            logging.warning(f"Track ID '{track_id}' not found in Word2Vec model's vocabulary.")
            return []
        similar_tracks = model.wv.most_similar(track_id, topn=top_n)
    elif isinstance(model, Doc2Vec):
        if track_id not in model.dv:
            logging.warning(f"Track ID '{track_id}' not found in Doc2Vec model's vocabulary.")
            return []
        similar_tracks = model.dv.most_similar(track_id, topn=top_n)
    else:
        raise ValueError("Invalid model type")
    
    return [track[0] for track in similar_tracks]


def get_recommendations_for_playlist(model, playlist_id, test_set, top_n=10):
    tracklist = test_set.loc[test_set['playlist_id'] == playlist_id, 'track_id'].values[0]
    avg_vector = mean_vectors(tracklist, model)  # Calculate the average vector for the playlist
    if avg_vector is None:
        return []
    predicted_tracks = model.wv.similar_by_vector(avg_vector, topn=top_n)  # Get similar tracks

    return [track for track, _ in predicted_tracks]  # Return just track IDs


def calculate_metrics(ground_truth, predictions):
    y_true = []
    y_pred = []

    for _, gt_row in ground_truth.iterrows():
        playlist_id = gt_row['playlist_id']
        actual_tracks = gt_row['actual_track_ids']
        
        pred_row = predictions[predictions['playlist_id'] == playlist_id]
        predicted_tracks = pred_row['predicted_track_ids'].values[0] if not pred_row.empty else []
        
        y_true.extend(actual_tracks)
        y_pred.extend(predicted_tracks)

    lb = LabelBinarizer()
    y_true_bin = lb.fit_transform(y_true)
    y_pred_bin = lb.fit_transform(y_pred)

    precision = precision_score(y_true_bin, y_pred_bin, average='macro')
    recall = recall_score(y_true_bin, y_pred_bin, average='macro')
    f1 = f1_score(y_true_bin, y_pred_bin, average='macro')
    accuracy = accuracy_score(y_true_bin, y_pred_bin, average='macro')
    r_precision = r_precision_score(predictions, ground_truth)
    ndcg = ndcg_score(predictions, ground_truth)

    return precision, recall, f1, accuracy, r_precision, ndcg


def r_precision_score(predictions, ground_truth):
    r_precisions = []
    for _, true_row in ground_truth.iterrows():
        playlist_id = true_row['playlist_id']
        true_tracks = true_row['actual_track_ids']
        pred_tracks = predictions[predictions['playlist_id'] == playlist_id]['predicted_track_ids'].values[0]

        R = len(true_tracks)
        relevant_retrieved = len(set(pred_tracks[:R]) & set(true_tracks))
        r_precisions.append(relevant_retrieved / R if R > 0 else 0)

    return sum(r_precisions) / len(r_precisions) if r_precisions else 0


def ndcg_score(predictions, ground_truth):
    ndcgs = []
    for _, true_row in ground_truth.iterrows():
        playlist_id = true_row['playlist_id']
        true_tracks = true_row['actual_track_ids']
        pred_tracks = predictions[predictions['playlist_id'] == playlist_id]['predicted_track_ids'].values[0]

        dcg = sum(1 / np.log2(i + 2) for i, track in enumerate(pred_tracks) if track in true_tracks)
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(true_tracks), len(pred_tracks))))

        ndcgs.append(dcg / idcg if idcg > 0 else 0)

    return sum(ndcgs) / len(ndcgs) if ndcgs else 0


def build_ground_truth(df):
    ground_truth = []

    for _, row in df.iterrows():
        playlist_id = row['playlist_id']
        tracklist = row['track_id']
        ground_truth.append((playlist_id, tracklist))
    
    return pd.DataFrame(ground_truth, columns=['playlist_id', 'actual_track_ids'])


def load_model(model_type):
    if model_type == 'D2V':
        return Doc2Vec.load(MDL_FOLDER + 'd2v/d2v-trained-model.model')
    elif model_type == 'W2V':
        return Word2Vec.load(MDL_FOLDER + 'w2v/w2v-trained-model.model')
    else:
        raise ValueError("Invalid model type, use: 'W2V' or 'D2V'")


def leave_one_out_evaluation(model, test_set, top_n):
    total_hits = 0
    total_playlists = 0

    for _, row in test_set.iterrows():
        playlist_id = row['playlist_id']
        tracklist = row['track_id']
        hits = 0
        
        for i, song in enumerate(tracklist):
            remaining_tracks = tracklist[:i].tolist() + tracklist[i + 1:].tolist()
            recommendations = get_recommendations_for_playlist(model, playlist_id, remaining_tracks, top_n)
            
            if song in recommendations:
                hits += 1
        
        total_hits += hits
        total_playlists += len(tracklist)

    average_hit_rate = total_hits / total_playlists if total_playlists > 0 else 0
    return average_hit_rate


def main(model_type, playlist_id=None, track_id=None):
    model = load_model(model_type)
    logging.info(f"Model loaded: {model_type}")
    
    test_set = pd.read_feather(SRC_FOLDER + 'test.feather')

    if playlist_id is not None:
        logging.info(f"Generating recommendations for playlist ID: {playlist_id}")
        recommendations = get_recommendations_for_playlist(model, playlist_id, test_set, top_n=10)
        logging.info(f"Recommended tracks for playlist {playlist_id}: {recommendations}")
    elif track_id is not None:
        logging.info(f"Finding similar tracks for track ID: {track_id}")
        similar_tracks = get_similar_tracks(model, track_id, top_n=10)
        logging.info(f"Similar tracks to {track_id}: {similar_tracks}")
    else:
        logging.error("Please provide either a playlist_id or a track_id for recommendations.")
        return

    ground_truth = build_ground_truth(test_set)

    precision, recall, f1, accuracy, r_precision, ndcg = calculate_metrics(test_set, ground_truth)
    logging.info(f'Accuracy: {accuracy:.4f}')
    logging.info(f'Precision: {precision:.4f}')
    logging.info(f'Recall: {recall:.4f}')
    logging.info(f'R-Precision: {r_precision:.4f}')
    logging.info(f'NDCG: {ndcg:.4f}')

    average_hit_rate = leave_one_out_evaluation(model, test_set, 10)
    logging.info(f'Average Hit Rate at 10: {average_hit_rate:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get recommendations for a playlist or similar tracks for a given track.')
    parser.add_argument('--use-model', type=str, required=True, choices=['W2V', 'D2V'], help='Specify the model to use: \'W2V\' or \'D2V\'')
    parser.add_argument('--playlist-id', type=str, help='Specify a playlist ID to get recommendations')
    parser.add_argument('--track-id', type=str, help='Specify a track ID to find similar tracks')

    args = parser.parse_args()
    main(args.use_model, playlist_id=args.playlist_id, track_id=args.track_id)
