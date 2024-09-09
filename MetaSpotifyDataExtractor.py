#little scripts for meta dati of spotify tracks
import requests # type: ignore

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
client_id = ''
client_secret = ''
token = get_access_token(client_id, client_secret)

if token:
    print(f"Access Token: {token}")

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

# Utilizzo:
track_id = ''
tracks = get_tracks(token, track_id)

if tracks:
    print(tracks)
