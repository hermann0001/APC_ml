#little scripts for meta dati of spotify tracks
import requests # type: ignore
import spotipy as sp
from spotipy import SpotifyClientCredentials
from utils import reconstruct_uri
import argparse
import sys

CLIENT_ID = '***'
CLIENT_SECRET = '***'
SRC_FOLDER = 'formatted/dataset/'
TRACKS_CSV = 'tracks.csv'

def get_spotify_metadata(uri, type):
    # initialize spotify api
    ccm = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
    api = sp.Spotify(client_credentials_manager=ccm)

    uri = reconstruct_uri(uri, type)

    try:
        if type == 'track':
            metadata = api.track(uri) 
        elif type == 'artist':
            metadata = api.artist(uri)
        else:
            metadata = api.album(uri)
        
        return metadata
    except Exception as e:
        print(f"Error retrieving metadata: {e}")
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Spoitify API Credentials")
    parser.add_argument('--client-id', type=str, required=True, help='Your Spotify Client ID')
    parser.add_argument('--client-secret', type=str, required=True, help='Your Spotify Client Secret')
    args = parser.parse_args()

    if not args.client_id or not args.client_secret:
        print("Usage: python3 MetaSpotifyDataExtractor.py --client-id='YOUR_CLIENT_ID' --client-secret='YOUR_CLIENT_SECRET'")
        sys.exit(1)

    CLIENT_ID = args.client_id
    CLIENT_SECRET = args.client_secret

    test_uri = "40Yq4vzPs9VNUrIBG5Jr2i"
    data = get_spotify_metadata(test_uri, 'artist')
    print(data)
