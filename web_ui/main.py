import json
import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import PIL.Image
import requests
import streamlit as st

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

HEADERS = {
    "accept": "application/json",
    "Content-Type": "application/json",
}
MAX_WIDTH: int = 2000
MAX_HEIGHT: int = 2000


def show_results(response: Dict[str, Any]) -> None:
    """Shows the predictions results to the ui
    :param response: the api response got from the model"""
    logging.error('SHOWING RESULTS\n\n\n')
    img: np.ndarray = response['image']
    pred: List[pd.DataFrame, str] = response['prediction']

    if not pred:
        st.error('No face detected, retry with a new image')
    else:
        n_faces: int = len(pred)
        face_string: str = 'face' if n_faces == 1 else 'faces'
        st.success(f"{n_faces} {face_string} detected, here's the result for each one of them")
        if img is not None:
            st.image(np.asarray(json.loads(img)))
        for i, (df, color) in enumerate(pred, start=1):
            df = pd.DataFrame.from_dict(json.loads(df)).reset_index(drop=True)
            st.markdown(f'### Detected face # {i} outlined in {color}. Predicted age estimation -> {df.loc[0].Age}')
            st.markdown(f'#### Top 5 predictions for face # {i}')
            st.dataframe(df)
            st.markdown('---')
    return


def send_api_request(img: np.ndarray) -> pd.DataFrame:
    """
    Sends the api requests with the given inputs and returns the response data formatted as a pandas dataframe. If
    response is an error, returns an empty dataframe
    :param img:
    :return:
    """
    encoded_image: str = json.dumps(img.tolist())
    response: requests.Response = requests.post(url="http://model_api:8000/predict/",
                                                json={'image': encoded_image},
                                                headers=HEADERS, verify=False)
    if response.ok:
        return response.json()
    else:
        st.error(response.text)
        return {'image': None, 'prediction': []}


def resize_image_if_too_big(img: np.ndarray) -> np.ndarray:
    w, h, c = img.shape

    if w < MAX_WIDTH and h < MAX_HEIGHT:
        return img

    aspect_ratio: float = w / h

    if w > h:
        new_width: int = MAX_WIDTH
        new_height: int = round(new_width // aspect_ratio)
    else:
        new_height: int = MAX_HEIGHT
        new_width: int = round(new_height * aspect_ratio)
    logging.error(img.shape)
    logging.error(f'{new_width} {new_height}')
    return np.array(PIL.Image.fromarray(img).resize((new_width, new_height)))


if __name__ == '__main__':
    col1, col2 = st.columns(2)
    image_upload = col1.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
    camera_photo = col2.camera_input("Take a picture")
    images_to_predict = []

    if image_upload is not None:
        images_to_predict.append(image_upload)
        col1.success("Successfully uploaded photo")
    if camera_photo is not None:
        images_to_predict.append(camera_photo)
        col1.success("Successfully taken photo")

    for image_to_predict in images_to_predict:
        image = PIL.Image.open(image_to_predict)
        image = np.array(image)

        image = resize_image_if_too_big(image)

        logging.error(f"image shape {image.shape}")
        prediction = send_api_request(image)
        show_results(prediction)
