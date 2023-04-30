import logging
from functools import lru_cache

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from api.age_estimator import AgeEstimator

logger = logging.getLogger()
logger.setLevel(logging.ERROR)


@lru_cache
def load_model() -> AgeEstimator:
    face_detector_model: AgeEstimator = AgeEstimator()
    return face_detector_model


class Image(BaseModel):
    image: np.ndarray

    class Config:
        arbitrary_types_allowed = True


app = FastAPI()


@app.post('/predict/')
def get_data(inputs: Image):
    model = load_model()
    return model.predict_from_image(inputs.image)


@app.get('/health')
def get_data():
    return {'health': 'ok'}
