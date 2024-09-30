import pandas as pd
import numpy as np
import argparse
from gensim.models import Doc2Vec, Word2Vec
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

MDL_FOLDER = 'models/'
SRC_FOLDER = 'formatted/dataset/'

def get_recommendations(model, test_set, top_n=10):    
    predictions = {}
    
    for index, row in test_set.iterrows():
        playlist_id = row['playlist_id']
        tracklist = row['track_id']
        
        predicted_tracks = []

        for track in tracklist:
            if isinstance(model, Word2Vec):
                similar_tracks = model.wv.most_similar(track, topn=top_n)
                predicted_tracks.extend([t[0] for t in similar_tracks])
            elif isinstance(model, Doc2Vec):
                similar_tracks = model.dv.most_similar(track, top_n)
                predicted_tracks.extend([t[0] for t in similar_tracks])

        predictions[playlist_id] = list(set(predicted_tracks))

    return predictions

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

    # Calculate metrics
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    accuracy = accuracy_score(y_true, y_pred, average='binary')
    r_precision = r_precision_score(predictions, ground_truth)
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
    
def main(model_type):
    model = load_model(model_type)
    test_set = pd.read_feather(SRC_FOLDER + 'test.feather')
    ground_truth = build_ground_truth(test_set)
    predictions = get_recommendations(model, test_set, top_n=10)
    precision, recall, f1, accuracy, r_precision, ndcg = calculate_metrics(test_set, ground_truth)
    print(f'Accuracy:    {accuracy:.4f}')
    print(f'Precision:   {precision:.4f}')
    print(f'Recall:      {recall:.4f}')
    print(f'R-Precision: {r_precision:.4f}')
    print(f'NDCG:        {ndcg:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load and evaluate model')
    parser.add_argument('--use-model', type=str, required=True, choices=['W2V', 'D2V'], help='Specify the model to use: \'W2V\' or \'D2V\'')
    args = parser.parse_args()
    main(args.use_model)

