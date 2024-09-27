from sklearn.model_selection import train_test_split
import pandas as pd

SRC_FOLDER = 'formatted/dataset/'

dataframe = pd.read_feather(SRC_FOLDER + 'dataframe.feather')

print("\nSplitting train and test set...")
playlist_documents = dataframe.groupby('playlist_id').agg(
    {
        'track_id': lambda x: list(x),
        'artist_id': lambda x: list(x),
        'pos': lambda x: list(x)
    }
).reset_index()

print(playlist_documents)

train, test = train_test_split(playlist_documents, test_size=0.2, random_state=666)

train.to_feather(SRC_FOLDER + "train.feather")
test.to_feather(SRC_FOLDER + "test.feather")