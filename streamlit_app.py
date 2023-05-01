import json
import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import PIL.Image
import streamlit as st

from api.age_estimator import AgeEstimator

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

MAX_WIDTH: int = 2000
MAX_HEIGHT: int = 2000

st.set_page_config(page_title='Age Estimator', page_icon=':boy:', layout='wide')


@st.cache_resource
def load_model() -> AgeEstimator:
    return AgeEstimator()


def show_results(response: Dict[str, Any]) -> None:
    """Shows the predictions results to the ui
    :param response: the api response got from the model"""
    img: np.ndarray = response['image']
    pred: List[pd.DataFrame, str] = response['prediction']

    if not pred:
        st.error('No face detected, retry with a new image')
    else:
        n_faces: int = len(pred)
        face_string: str = 'face' if n_faces == 1 else 'faces'
        with st.container():
            st.write('---')
            st.success(f"{n_faces} {face_string} detected, here's the result for each one of them",
                       icon='✅')
        if img is not None:
            st.image(img)
        for i, (df, color) in enumerate(pred, start=1):
            df = pd.DataFrame.from_dict(json.loads(df)).reset_index(drop=True)
            st.markdown(f'### Detected face outlined in {color}, predicted age: *{df.loc[0].Age}*')
            st.markdown('##### Top 5 predictions')
            st.dataframe(df)
            st.write('---')
    return


def resize_image_if_too_big(img: np.ndarray) -> np.ndarray:
    w, h, c = img.shape

    if w < MAX_WIDTH and h < MAX_HEIGHT:
        return img

    aspect_ratio: float = w / h

    if aspect_ratio > 1:
        new_width: int = MAX_WIDTH
        new_height: int = round(new_width // aspect_ratio)
    else:
        new_height: int = MAX_HEIGHT
        new_width: int = round(new_height * aspect_ratio)
    return np.array(PIL.Image.fromarray(img).resize((new_width, new_height)))


if __name__ == '__main__':
    model: AgeEstimator = load_model()
    with st.container():
        st.title('Age Estimator UI')
        st.subheader('In this page you can play with the age estimator app, load a picture from the disk or take a '
                     'picture using the camera, and in few seconds you\'ll get your estimated age')
        st.write('Please upload a clear picture of you face, or the model will not recognize the face and it will not '
                 'work. You can upload a group page, and it will predict the age for all the faces in the image')
        st.write('[Click here](https://github.com/paolo-sofia/age-estimation-project) to find about more')

    with st.container():
        st.write('---')
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
        prediction = model.predict_from_image(image)
        show_results(prediction)
