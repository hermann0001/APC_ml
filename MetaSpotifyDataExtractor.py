#little scripts for meta dati of spotify tracks
import requests # type: ignore
import spotipy as sp
import pandas as pd
import os
from utils import reconstruct_uri

CLIENT_ID = ''
CLIENT_SECRET = ''
SRC_FOLDER = 'formatted/dataset/'
TRACKS_CSV = 'tracks.csv'

def get_access_token(client_id, client_secret):
    url = 'https://accounts.spotify.com/api/token'
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    body = {
        'grant_type': 'client_credentials',
        'client_id': client_id,
        'client_secret': client_secret
    }

    response = requests.post(url, headers=headers, data=body)
    
    if response.status_code == 200:
        return response.json().get('access_token')
    else:
        print(f"Error: {response.status_code}")
        return None

# Utilizzo:


def get_tracks(access_token, track):
    url = 'https://api.spotify.com/v1/audio-features/' + track
    headers = {
        'Authorization': f'Bearer {access_token}'
    }

    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None


if __name__ == '__main__':
    
    # token = get_access_token(CLIENT_ID, CLIENT_SECRET)

    # if token:
    #     print(f"Access Token: {token}")

    # track_id = ''
    # tracks = get_tracks(token, track_id)

    # if tracks:
    #     print(tracks)
    
    tracks_data = pd.read_csv(os.path.join(SRC_FOLDER, TRACKS_CSV), usecols=['track_uri'])
    tracks_data['track_id'] = tracks_data['track_uri']
    tracks_data['track_uri'] = tracks_data['track_uri'].apply(reconstruct_uri)
    tracks_data.drop_duplicates(inplace=True)
    print(tracks_data.head())