import pandas as pd


PLAYLIST_CSV = "formatted/dataset/playlists.csv"
PLAYLIST_TEST_CSV = "formatted/challenge_set/playlists_tracks.csv"

playlists_df = pd.read_csv(PLAYLIST_CSV, usecols=['playlist_id', 'name'])
playlists_tracks_test_df = pd.read_csv(PLAYLIST_TEST_CSV, usecols=['playlist_id', 'track_id'])

print(f"playlists_df duplicated rows: {playlists_df.duplicated().sum()}")
print(f"playlists_test_df duplicated rows: {playlists_tracks_test_df.duplicated().sum()}")


playlists_df = playlists_df.drop_duplicates().reset_index(drop=True)
playlists_test_df = playlists_tracks_test_df.drop_duplicates().reset_index(drop=True)

common_ids = playlists_tracks_test_df['playlist_id'].isin(playlists_df['playlist_id']).sum()
total_ids = len(playlists_tracks_test_df)
print(f"{common_ids} playlist_id su {total_ids} trovati in playlists_df")

print(f"playlists_df duplicated rows: {playlists_df.duplicated().sum()}")


merged_df = pd.merge(playlists_df,playlists_tracks_test_df, on='playlist_id', how='left')

print(merged_df.head())