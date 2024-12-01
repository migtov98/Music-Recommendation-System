import json

import requests
import urllib.parse
import search_song as ss
import pulse_recommender as recommender

from dotenv import load_dotenv
import os

from datetime import datetime, timedelta
from flask import Flask, redirect, url_for, render_template, request, jsonify, session, flash, make_response

#Applying Secrets
load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("APP_SECRET_KEY")
app.permanent_session_lifetime = timedelta(minutes=15)

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")

REDIRECT_URI = "http://localhost:5000/callback"
AUTH_URL = "https://accounts.spotify.com/authorize"
TOKEN_URL = "https://accounts.spotify.com/api/token"
API_BASE_URL = "https://api.spotify.com/v1/"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/login")
def login():
    scope = "playlist-read-private user-read-private user-read-email"

    if "access_token" not in session:
        params = {
            "client_id": CLIENT_ID,
            "response_type": "code",
            "scope": scope,
            "redirect_uri": REDIRECT_URI,
            "show_dialog": True,  # Shows login screen
        }

        auth_url = f"{AUTH_URL}?{urllib.parse.urlencode(params)}"   
        flash("Login Successful!")  
        return redirect(auth_url)
    else:
        if "access_token" in session:
            flash("Already Logged In!")            
            return redirect(url_for("user"))    

@app.route("/logout")
def logout():
    flash("Logout Successful!", "info")
    #Removes data from session    
    session.pop("access_token",None) 
    session.pop("refresh_token",None) 
    session.pop("expires_at",None)
    session.pop("user_id",None) 
    session.pop("user_name",None) 
    session.pop("playlist_id",None)   
    session.pop("playlist_name",None)
    return redirect(url_for("home"))

@app.route("/callback")
def callback():
    if "error" in request.args:        
        return jsonify({"error": request.args["error"]})

    if "code" in request.args:
        req_body = {
            "code": request.args["code"],
            "grant_type": "authorization_code",
            "redirect_uri": REDIRECT_URI,
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        }

        response = requests.post(TOKEN_URL, data=req_body)
        token_info = response.json()

        session["access_token"] = token_info["access_token"]
        session["refresh_token"] = token_info["refresh_token"]
        # token only lasts 1 day
        session["expires_at"] = datetime.now().timestamp() + token_info["expires_in"]

        return redirect('/user')

@app.route("/get_user")
def get_user():
    if "access_token" not in session:
        return redirect("/login")

    if datetime.now().timestamp() > session["expires_at"]:
        return redirect("/refresh-token")
    
    headers = {"Authorization": f"Bearer {session['access_token']}"}

    response = requests.get(API_BASE_URL + "me", headers=headers)
    user = response.json()
    user_id = user["id"]
    user_name = user["display_name"]
    
    session["user_id"] = user_id
    session["user_name"] = user_name

    return user_name

@app.route("/get_playlists")
def get_playlists():
    if "access_token" not in session:
        return redirect("/login")

    if datetime.now().timestamp() > session["expires_at"]:
        return redirect("/refresh-token")
    
    headers = {"Authorization": f"Bearer {session['access_token']}"}

    try: 
        response = requests.get(API_BASE_URL + "me/playlists", headers=headers)
        playlists = response.json()

        playlist_dict = {}
        for i in playlists["items"]:
            playlist_dict[i["name"]] = i["id"]    

    except Exception as e:
        print(e)
    
    return playlist_dict

@app.route("/get_playlist_key")
def get_playlist_key(playlist_dict, playlist_id):
    for key, value in playlist_dict.items():
        if playlist_id == value:
            return key
    flash("ERROR: NO KEY FOUND")  
    return redirect("/user")

@app.route("/get_playlist_track_info")
def get_playlist_track_info():
    if "access_token" not in session:
        return redirect("/login")

    if datetime.now().timestamp() > session["expires_at"]:
        return redirect("/refresh-token")
    
    headers = {"Authorization": f"Bearer {session['access_token']}"}

    response = requests.get(API_BASE_URL + "playlists/" + session["playlist_id"] + "/tracks", headers=headers)
    playlist = response.json()   

    playlist_track_dict = {}
    for i in playlist["items"]:
        playlist_track_dict[i["track"]["id"]] = [i['track']['artists'][0]['name'], i["track"]["name"], i['added_at']]

    return playlist_track_dict

@app.route("/search", methods=["GET", "POST"])
def search():
    session.pop("artist_search",None)
    if request.method == "POST":
        artist = request.form["artist"]
        session["artist_search"] = artist                
        return redirect(url_for("view_search_results"))

    return render_template("search.html")

@app.route("/view_search_results", methods=["GET", "POST"])
def view_search_results():
    song_dict = ss.search_artist(session["artist_search"])    
    return render_template("view_search_results.html", tracks=song_dict, artist = session["artist_search"])

@app.route("/view_search_recommendations")
def view_search_recommendations():    
    artist = session["artist_search"]
    song_dict = ss.search_artist(artist)    
    song_recommendation_dict = recommender.main(song_dict)
    return render_template("view_search_recommendations.html", tracks=song_recommendation_dict, aritst=artist)

@app.route("/user", methods=['GET','POST'])
def user():  
    if request.method == "POST":
        playlist = request.form['playlist']
        session["playlist_id"] = playlist
        session["playlist_name"] = get_playlist_key(get_playlists(),session["playlist_id"])
        return redirect(url_for("view_playlist_songs"))  

    user = get_user()
    playlist_dict = get_playlists()
    return render_template("user.html", user=user, playlists=playlist_dict)    

@app.route("/view_playlist_songs", methods=['GET','POST'])
def view_playlist_songs():
    playlist_track_dict = get_playlist_track_info()
    song_dict = ss.get_dataset_songs_from_playlist(playlist_track_dict)
    return render_template("view_playlist_songs.html", tracks=song_dict, playlist_name=session["playlist_name"])   

@app.route("/view_playlist_recommendations") 
def view_playlist_recommendations():
    playlist_track_dict = get_playlist_track_info()
    song_dict = ss.get_dataset_songs_from_playlist(playlist_track_dict)
    song_recommendation_dict = recommender.main(song_dict)
    return render_template("view_playlist_recommendations.html", tracks=song_recommendation_dict, playlist=session["playlist_name"])


@app.route("/refresh-token")
def refresh_token():
    if "refresh_token" not in session:
        return redirect("/login")

    if datetime.now().timestamp() > session["expirest_at"]:
        req_body = {
            "grant_type": "refresh_token",
            "refresh_token": session["refresh_token"],
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        }

        response = requests.post(TOKEN_URL, data=req_body)
        new_token_info = response.json()

        session["access_token"] = new_token_info["access_token"]
        session["expires_at"] = (
            datetime.now().timestamp() + new_token_info["expires_in"]
        )

        return redirect("/user")

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
