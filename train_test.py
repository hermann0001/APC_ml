from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

SAMPLE_SIZE = 100000 # no. of rows loaded (total = 1M)
CSV = "temp/combined_playlists.csv" 

df = pd.read_csv(CSV, nrows=SAMPLE_SIZE)


# Split into initial training+validation and test sets
train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Split the training+validation set into training and validation sets
train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42)

track_counts = train_df['track_uri'].value_counts()
min_track_occurrences = track_counts.min()
print(f"Minimum number of track occurrences: {min_track_occurrences}")
