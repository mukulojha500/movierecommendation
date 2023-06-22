import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json
import zipfile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the preprocessed data and create the movie predictor object
zip_file_path = r"mukulojha500/movierecommendation/tmdb_5000_credits.zip"
file_name = "tmdb_5000_credits.csv"
credits
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        credits=pd.read_csv(file_name)
movies = pd.read_csv('tmdb_5000_movies.csv')
movies = movies.merge(credits, on='title')
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
movies.dropna(inplace=True)
movies.duplicated().sum()

import ast

def convert(obj):
    l = []
    for i in ast.literal_eval(obj):
        l.append(i['name'])
    return l

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

def convert3(obj):
    l = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            l.append(i['name'])
            counter += 1
        else:
            break
    return l

movies['cast'] = movies['cast'].apply(convert3)

def fetch_director(obj):
    l = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            l.append(i['name'])
    return l

movies['crew'] = movies['crew'].apply(fetch_director)
movies['overview'] = movies['overview'].apply(lambda x: x.split())
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
new_df = movies[['movie_id', 'title', 'tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_df['tags'] = new_df['tags'].apply(stem)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
similarity = cosine_similarity(vectors)

def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    predictions = [new_df.iloc[i[0]].title for i in movies_list]
    return predictions

# Create the MoviePredictor class

class MoviePredictor:
    def __init__(self, new_df, cv, recommend):
        self.new_df = new_df
        self.cv = cv
        self.recommend = recommend

# Create an instance of the MoviePredictor class
movie_predictor = MoviePredictor(new_df, cv, recommend)

# Pickle the movie_predictor object
filename = 'movie_predictor.sav'
with open(filename, 'wb') as file:
    pickle.dump(movie_predictor, file)

model = pickle.load(open('movie_predictor.sav', 'rb'))

class ModelInput(BaseModel):
    movie: str

@app.post('/movie_recommendation')
def movie_recommend(input_parameters: ModelInput):
    movie = input_parameters.movie
    predictions = model.recommend(movie)
    return {'predictions': predictions}    
