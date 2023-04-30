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


def show_results(response: Dict[str, Any]) -> None:
    img: np.ndarray = response['image']
    pred: List[pd.DataFrame, str] = response['prediction']

    if not pred:
        st.error('No face detected, retry with a new image')
    else:
        n_faces: int = len(pred)
        face_string: str = 'face' if n_faces == 1 else 'faces'
        st.success(f"{n_faces} {face_string} detected, here's the result for each one of them")
        st.image(img)
        for i, (df, color) in enumerate(pred, start=1):
            st.markdown(f'## Detected face # {i} outlined in {color}. Preicted age estimation -> {df.loc[0].Age}')
            st.markdown(f'#### Top 5 predictions for face # {i}')
            st.dataframe(df)
    return


def send_api_request(img: np.ndarray) -> pd.DataFrame:
    """
    Sends the api requests with the given inputs and returns the response data formatted as a pandas dataframe. If
    response is an error, returns an empty dataframe
    :param img:
    :return:
    """
    response: requests.Response = requests.post(url="http://spark_api:8000/predict/", json={'image': img},
                                                headers=HEADERS, verify=False)
    if response.ok:
        st.success("Successfully computed data")
        try:
            return pd.DataFrame().from_dict(response.json())
        except Exception as e:
            logger.error(e)
            return pd.DataFrame()
    else:
        st.error(response.text)
        return pd.DataFrame()


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
        prediction = send_api_request(image)
        show_results(prediction)
