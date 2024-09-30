import pandas as pd
import numpy as np
import argparse
from gensim.models import Doc2Vec, Word2Vec
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import LabelBinarizer
from collections import OrderedDict


MDL_FOLDER = 'models/'
SRC_FOLDER = 'formatted/dataset/'

def get_similar_tracks(model, track_id, top_n=10):
    if isinstance(model, Word2Vec):
        if track_id not in model.wv:
            print(f"Warning: Track ID '{track_id}' not found in Word2Vec model's vocabulary.")
            return []
        similar_tracks = model.wv.most_similar(track_id, topn=top_n)
    elif isinstance(model, Doc2Vec):
        if track_id not in model.dv:
            print(f"Warning: Track ID '{track_id}' not found in Doc2Vec model's vocabulary.")
            return []
        similar_tracks = model.dv.most_similar(track_id, topn=top_n)
    else:
        raise ValueError("Invalid model type")
    
    return [track[0] for track in similar_tracks]

def get_similar_tracks_by_id(model, track_id, top_n=10):
    similar_tracks = []
    
    # Search in the model's vocabulary for tokens containing the track_id
    for token in model.wv.index_to_key:
        if token.startswith(f"{track_id}_"):
            # Get most similar tracks for the found token
            most_similar = model.wv.most_similar(token, topn=top_n)
            
            # Extract just the track_id from the similar tokens
            similar_tracks.extend([sim_track.split('_')[0] for sim_track, _ in most_similar])
    
    # Remove duplicates while preserving order
    similar_tracks = list(OrderedDict.fromkeys(similar_tracks))
    
    return similar_tracks

def get_recommendations_for_playlist(model, playlist_id, test_set, top_n=10):
    tracklist = test_set.loc[test_set['playlist_id'] == playlist_id, 'track_id'].values[0]
    predicted_tracks = []

    for track in tracklist:
        similar_tracks = get_similar_tracks(model, track, top_n)
        predicted_tracks.extend(similar_tracks)

    # Remove duplicates while preserving order
    return list(OrderedDict.fromkeys(predicted_tracks))

def calculate_metrics(ground_truth, predictions):
    # Initialize lists to hold true labels and predictions
    y_true = []
    y_pred = []
    
    for _, gt_row in ground_truth.iterrows():
        playlist_id = gt_row['playlist_id']
        actual_tracks = gt_row['actual_track_ids']
        
        # Find the corresponding predicted tracks
        pred_row = predictions[predictions['playlist_id'] == playlist_id]
        predicted_tracks = pred_row['predicted_track_ids'].values[0] if not pred_row.empty else []
        
        # Extend true and predicted lists
        y_true.extend(actual_tracks)
        y_pred.extend(predicted_tracks)

    lb = LabelBinarizer()
    y_true_bin = lb.fit_transform(y_true)
    y_pred_bin = lb.fit_transform(y_pred)
    # Calculate metrics
    print("Calculating precision")
    precision = precision_score(y_true_bin, y_pred_bin, average='macro')
    print("Calculating recall")
    recall = recall_score(y_true_bin, y_pred_bin, average='macro')
    print("Calculating f1 score")
    f1 = f1_score(y_true_bin, y_pred_bin, average='macro')
    print("Calculating accuracy")
    accuracy = accuracy_score(y_true_bin, y_pred_bin, average='macro')
    print("Calculating r_precision")
    r_precision = r_precision_score(predictions, ground_truth)
    print("Calculating ndcg")
    ndcg = ndcg_score(predictions, ground_truth)


    return precision, recall, f1, accuracy, r_precision, ndcg

def get_songs(playlist_id, test_set):
    track_list = test_set.at[playlist_id, 'track_id']
    artist_list = test_set.at[playlist_id, 'artist_is']
    pos_list = test_set.at[playlist_id, 'pos']

    return track_list, artist_list, pos_list

# R-Precision is defined as the proportion of relevant items retrieved in the top R results, where 
# R is the number of relevant items for a given query (in this case, the number of true track IDs in the ground truth).
def r_precision_score(predictions, ground_truth):
    r_precisions = []
    for playlist_id, true_tracks, in ground_truth:
        pred_tracks = predictions[playlist_id]

        R = len(true_tracks)

        relevant_retrieved = len(set(pred_tracks[:R]) & set(true_tracks))

        r_precisions.append(relevant_retrieved / R if R > 0 else 0)
    return sum(r_precisions) / len(r_precisions)

def ndcg_score(predictions, ground_truth):
    ndcgs = []
    for playlist_id, true_tracks in ground_truth:
        pred_tracks = predictions[playlist_id]

        dcg = sum(1 / np.log2(i + 2) for i, track in enumerate(pred_tracks) if track in true_tracks)
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(true_tracks), len(pred_tracks))))

        ndcgs.append(dcg / idcg if idcg > 0 else 0)
    return sum(ndcgs) / len(ndcgs)

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
    
def main(model_type, playlist_id=None, track_id=None):
    model = load_model(model_type)
    print(f"model loaded with params: {model}")
    test_set = pd.read_feather(SRC_FOLDER + 'test.feather')

    if playlist_id is not None:
        print(f"Generating recommendations for playlist ID: {playlist_id}")
        #print_playlist_info(playlist_id)
        recommendations = get_recommendations_for_playlist(model, playlist_id, test_set, top_n=10)
        print(f"Recommended tracks for playlist {playlist_id}: {recommendations}")
    elif track_id is not None:
        print(f"Finding similar tracks for track ID: {track_id}")
        #print_track_info(track_id)
        similar_tracks = get_similar_tracks(model, track_id, top_n=10)
        print(f"Similar tracks to {track_id}: {similar_tracks}")
    else:
        print("Please provide either a playlist_id or a track_id for recommendations.")


    print("\ntest set:")
    print(test_set)
    ground_truth = build_ground_truth(test_set)
    print("\nground truth:")
    print(ground_truth)

    precision, recall, f1, accuracy, r_precision, ndcg = calculate_metrics(test_set, ground_truth)
    print(f'Accuracy:    {accuracy:.4f}')
    print(f'Precision:   {precision:.4f}')
    print(f'Recall:      {recall:.4f}')
    print(f'R-Precision: {r_precision:.4f}')
    print(f'NDCG:        {ndcg:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get recommendations for a playlist or similar tracks for a given track.')
    parser.add_argument('--use-model', type=str, required=True, choices=['W2V', 'D2V'], help='Specify the model to use: \'W2V\' or \'D2V\'')
    parser.add_argument('--playlist-id', type=str, help='Specify a playlist ID to get recommendations')
    parser.add_argument('--track-id', type=str, help='Specify a track ID to find similar tracks')

    args = parser.parse_args()
    main(args.use_model, playlist_id=args.playlist_id, track_id=args.track_id)


# pid           21058
# track_id      39338