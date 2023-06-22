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
zip_file_path = r"./tmdb_5000_credits.zip"
file_name = "tmdb_5000_credits.csv"

# Load movies and credits data as generators
movies = pd.read_csv('tmdb_5000_movies.csv', iterator=True)
credits = pd.read_csv(zip_file_path, compression='zip', iterator=True)
    
chunk_size = 1000  # Number of rows to process per chunk
movies_chunk = movies.get_chunk(chunk_size)
credits_chunk = credits.get_chunk(chunk_size)

movies = movies_chunk.merge(credits_chunk, on='title')
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies.dropna(inplace=True)
movies.duplicated().sum()

import ast
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def convert(obj):
    l = []
    for i in ast.literal_eval(obj):
        l.append(i['name'])
    return l

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

def fetch_director(obj):
    l = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            l.append(i['name'])
    return l

# Initialize stemmer and vectorizer
ps = PorterStemmer()
cv = CountVectorizer(max_features=5000, stop_words='english')
def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)
def preprocess_data(chunk):
    chunk['genres'] = chunk['genres'].apply(convert)
    chunk['keywords'] = chunk['keywords'].apply(convert)
    chunk['cast'] = chunk['cast'].apply(convert3)
    chunk['crew'] = chunk['crew'].apply(fetch_director)
    chunk['overview'] = chunk['overview'].apply(lambda x: x.split())
    chunk['genres'] = chunk['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
    chunk['keywords'] = chunk['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
    chunk['cast'] = chunk['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
    chunk['crew'] = chunk['crew'].apply(lambda x: [i.replace(" ", "") for i in x])
    chunk['tags'] = chunk['overview'] + chunk['genres'] + chunk['keywords'] + chunk['cast'] + chunk['crew']
    chunk[['movie_id', 'title', 'tags']] = chunk[['movie_id', 'title', 'tags']].astype(str)
    chunk['tags'] = chunk['tags'].apply(lambda x: " ".join(x))
    chunk['tags'] = chunk['tags'].apply(lambda x: x.lower())
    chunk['tags'] = chunk['tags'].apply(stem)
    vectors = cv.fit_transform(chunk['tags']).toarray()
    similarity = cosine_similarity(vectors)
    return chunk, similarity

movies, similarity = preprocess_data(movies)

# Create the MoviePredictor class
class MoviePredictor:
    def __init__(self, movies, similarity):
        self.movies = movies
        self.similarity = similarity
    
    def recommend(self, movie):
        movie_index = self.movies[self.movies['title'] == movie].index[0]
        distances = self.similarity[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        predictions = [self.movies.iloc[i[0]].title for i in movies_list]
        return predictions

# Create an instance of the MoviePredictor class
movie_predictor = MoviePredictor(movies, similarity)

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
