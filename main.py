from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json
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

model = pickle.load(open('movie_predictor.sav', 'rb'))

class ModelInput(BaseModel):
    movie: str

@app.post('/movie_recommendation')
def movie_recommend(input_parameters: ModelInput):
    movie = input_parameters.movie
    predictions = model.recommend(movie)
    return {'predictions': predictions}
