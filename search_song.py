import pandas as pd
import datetime

music_df = pd.read_csv("spotify_dataset_v2.csv")

#Retrieves artist's songs that are in our dataset
def search_artist(artist):    
    results = music_df[music_df['artist'] == artist]
    
    song_dict = {}
    items = results["track_id"].size
    idx = 0
    current_time = datetime.datetime.now()
    for i in range(items):
        track_id = results["track_id"].iloc[idx]
        track_name = results["track_name"].iloc[idx]
        artist = results["artist"].iloc[idx]        
        song_dict[track_id] = [track_name, artist, current_time]
        
        idx+=1

    if len(song_dict)==0:
        return "No artist of that name in our database. Please try a different artist."
    else:
        return song_dict
    
#This will only get playlist songs that are available in our dataset
def get_dataset_songs_from_playlist(playlist_track_dict):
    song_dict = {}
    for id, track_info in playlist_track_dict.items():
        if id in music_df["track_id"].values: #Check if id is in dataset
            song_dict[id] = [track_info[0], track_info[1], track_info[2]]

    if len(song_dict)==0:
        return "There are no songs from this playlist in our database. Please try a different playlist."
    else:
        return song_dict