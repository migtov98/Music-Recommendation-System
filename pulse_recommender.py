import pandas as pd
import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

music_df = pd.read_csv("spotify_dataset_v2.csv")

################################ Preparing Data ################################

music_df = music_df.rename(columns={'track_id':'id','track_year':'year'})
music_df = music_df[['id', 'artist', 'track_name', 'year','artist_popularity','artist_genres','danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature','artist_and_track']]

index = 0
for row in music_df['artist_genres']:
    music_df.loc[index,'artist_genres_upd'] = row.replace(" ", "_")    
    index+=1

music_df['artist_genres_upd_list'] = music_df['artist_genres_upd'].apply(lambda x: x[0:music_df['artist_genres_upd'].size].split(',_'))
music_df['year'] = music_df['year'].astype('Int64')

music_df.drop_duplicates(subset=["id"],keep="first",inplace = True)
music_df.drop_duplicates(subset=["track_name","artist"],keep="first",inplace = True)

################################ Feature Engineering ################################

#Normalize float varialbes with MinMaxScaler
def normalize(music_df, float_cols):
    #scale float columns
    floats = music_df[float_cols].reset_index(drop = True) #These are the audio features that are float type
    scaler = MinMaxScaler() #sets all audio features values between 0 and 1
    floats_normalized = pd.DataFrame(scaler.fit_transform(floats), columns = floats.columns) * .35  #Put scaled data to new dataset
    return floats_normalized

#TFIDF Vectorizer on genre column
def tfidf_vectorize_genres(music_df,genre_column):    
    #tfidf Vectorize genre lists
    tfidf = TfidfVectorizer() #Creating instance of TfidfVectorizer 
    tfidf_matrix = tfidf.fit_transform(music_df[genre_column].apply(lambda x: " ".join(x))) #Creates a sparse matrix using genre elements
    genre_df = pd.DataFrame(tfidf_matrix.toarray()) #Creates a dataframe with the genres vectorized 
    genre_df.columns = ['genre' + "|" + i for i in tfidf.get_feature_names_out()] #Adds genre name to its respective column 
    genre_df.reset_index(drop = True, inplace=True)
    return genre_df

#One Hot Encoding
def ohe_prep(music_df, column, new_column_name):
    ohe_df = pd.get_dummies(music_df[column])                                # sets 0/1 indicator variables
    feature_names = ohe_df.columns                                           # types of values in column
    ohe_df.columns = [new_column_name + "_" + str(i) for i in feature_names] # creates new column with new name
    ohe_df.reset_index(drop = True, inplace = True)                          # resets index back to default index
    return ohe_df

################################ Create User Dataframe ################################

def create_user_dataframe(user_song_dict,df_features):
    df = pd.DataFrame() 
    idx = 0
    for id, track_info in user_song_dict.items():
        df.loc[idx,"id"] = id
        df.loc[idx,"artist"] = track_info[0]
        df.loc[idx,"track_name"] = track_info[1]
        df.loc[idx,"date_added"] = track_info[2]
        idx += 1
    df["date_added"] = pd.to_datetime(df['date_added'])
    df = df[df['id'].isin(df_features['id'].values)].sort_values('date_added',ascending = False) 
    return df

def check_date_added_column(user_df):
    if 'date_added' not in user_df.columns:
        current_time = datetime.datetime.now()
        size = user_df["id"].size
        for i in range(size):
            user_df.loc[i,"date_added"] = current_time
        return user_df
    else:
        return user_df

################################ Generate Recommendations ################################

def generate_playlist_feature(complete_feature_set, playlist_df, weight_factor):
    """ 
    Summarize a user's playlist into a single vector

    Parameters: 
        complete_feature_set (pandas dataframe): Dataframe which includes all of the features for the spotify songs
        playlist_df (pandas dataframe): playlist dataframe
        weight_factor (float): float value that represents the recency bias. The larger the recency bias, the most priority recent songs get. Value should be close to 1. 
        
    Returns: 
        playlist_feature_set_weighted_final (pandas series): single feature that summarizes the playlist
        complete_feature_set_nonplaylist (pandas dataframe): 
    """
    # feature set with playlist songs
    complete_feature_set_playlist = complete_feature_set[complete_feature_set['id'].isin(playlist_df['id'].values)]
    complete_feature_set_playlist = complete_feature_set_playlist.merge(playlist_df[['id','date_added']], on = 'id', how = 'inner')    
    
    # feature set without playlist songs
    complete_feature_set_nonplaylist = complete_feature_set[~complete_feature_set['id'].isin(playlist_df['id'].values)]
    
    # sort by the date added (weight)
    playlist_feature_set = complete_feature_set_playlist.sort_values('date_added',ascending=False)
    most_recent_date = playlist_feature_set.iloc[0,-1]
    
    # calculate the number of months between the most recent date
    for ix, row in playlist_feature_set.iterrows():
        playlist_feature_set.loc[ix,'months_from_recent'] = int((most_recent_date.to_pydatetime() - row.iloc[-1].to_pydatetime()).days / 30)
        
    # calculates the weight of each row by raising the "weigh factor" to the power of the negative months from the most recent date
    playlist_feature_set['weight'] = playlist_feature_set['months_from_recent'].apply(lambda x: weight_factor ** (-x))
    
    # multiplies all the columns except for the last 4 columns by the weight
    playlist_feature_set_weighted = playlist_feature_set.copy()    
    playlist_feature_set_weighted.update(playlist_feature_set_weighted.iloc[:,:-4].mul(playlist_feature_set_weighted.weight,0))
    playlist_feature_set_weighted_final = playlist_feature_set_weighted.iloc[:, :-4]    
    
    # returns the user's playlist by summing all of the columns to create a single vector
    return playlist_feature_set_weighted_final.sum(axis = 0), complete_feature_set_nonplaylist

def generate_playlist(df, features, nonplaylist_features):
    """ 
    Pull songs from a specific playlist.

    Parameters: 
        df (pandas dataframe): spotify dataframe
        features (pandas series): summarized playlist feature
        nonplaylist_features (pandas dataframe): feature set of songs that are not in the selected playlist
        
    Returns: 
        non_playlist_df_top_recommendations: Top recommendations for that playlist
    """
     
    non_playlist_df = df[df['id'].isin(nonplaylist_features['id'].values)]
    non_playlist_df['similarity'] = cosine_similarity(nonplaylist_features.drop('id', axis = 1).values, features.values.reshape(1, -1))[:,0]    
    non_playlist_df_top_recommendations = non_playlist_df.sort_values('similarity',ascending = False).head(100)    
    non_playlist_df_top_recommendations.drop_duplicates(subset=['id'],keep="first",inplace = True)
    
    return non_playlist_df_top_recommendations

def get_generated_playlist_dict(playlist_df):
    song_dict = {}
    items = playlist_df["id"].size
    idx = 0

    for i in range(items):
        track_id = playlist_df["id"].iloc[idx]
        artist = playlist_df["artist"].iloc[idx]
        track_name = playlist_df["track_name"].iloc[idx]   
        year = playlist_df["year"].iloc[idx]          
        similarity = playlist_df["similarity"].iloc[idx]  
        similarity = str(round(similarity, 2)) 
        song_dict[track_id] = [artist, track_name, year, similarity]
        idx+=1  

    return song_dict

################################ Main ################################

def main(user_song_dict):
    float_cols = music_df.dtypes[music_df.dtypes == 'float64'].index.values
    floats_normalized = normalize(music_df, float_cols)

    genres_vectorized = tfidf_vectorize_genres(music_df,"artist_genres_upd_list")

    #One Hot Encoding
    ohe_year = ohe_prep(music_df, "year", "year") * 0.45 #Closer to 1 = recommendation closer to year
    ohe_key = ohe_prep(music_df, "key", "key") * .98
    ohe_mode = ohe_prep(music_df, "mode", "mode") * .98

    music_df['popularity_red'] = music_df['artist_popularity'].apply(lambda x: int(x/5))
    ohe_popularity = ohe_prep(music_df, "popularity_red", "pop") * 0.18

    #Concatenate all features
    df_features = pd.concat([genres_vectorized, floats_normalized, ohe_popularity,ohe_year,ohe_key,ohe_mode], axis = 1)
    df_features['id']=music_df['id'].values

    #Create user's dataframe containing songs with feature vectors
    user_df = create_user_dataframe(user_song_dict,df_features)
    user_df = check_date_added_column(user_df)
    
    #User's feature vector
    complete_feature_set_playlist_vector, complete_feature_set_nonplaylist = generate_playlist_feature(df_features,user_df,1.02)

    #Creates dataframe with recommended songs and stores them in a dataframe
    generated_recommendations_df = generate_playlist(music_df, complete_feature_set_playlist_vector, complete_feature_set_nonplaylist)

    #Stores songs in dataframe to a dictionary
    recommended_songs_dict = get_generated_playlist_dict(generated_recommendations_df)

    return recommended_songs_dict