import json
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
    image: str

    class Config:
        arbitrary_types_allowed = True


model = load_model()
app = FastAPI()


@app.post('/predict/')
def get_data(inputs: Image):
    image = np.asarray(json.loads(inputs.image)).astype(np.uint8)
    response = model.predict_from_image(image)
    response['image'] = json.dumps(response['image'].tolist())
    return response


@app.get('/health')
def get_data():
    return {'health': 'ok'}
