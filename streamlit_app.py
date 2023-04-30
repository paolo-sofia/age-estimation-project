from typing import List, Tuple

import numpy as np
import pandas as pd
import PIL.Image
import streamlit as st
import tensorflow as tf
from PIL.ImageDraw import Draw
from tensorflow.python.framework.ops import convert_to_tensor

from detector import FaceDetector

tf.config.experimental.set_visible_devices(tf.config.list_physical_devices('CPU'))
COLORS: Tuple[str] = ('red', 'green', 'blue', 'yellow', 'cyano', 'orange')


@st.cache_resource()
def load_model() -> FaceDetector:
    """
    Loads the face detector model to be used for inference
    :return:
    """
    face_detector: FaceDetector = FaceDetector()
    return face_detector


def predict_from_image(img: np.ndarray) -> None:
    model: FaceDetector = load_model()

    faces, bboxes = model.fit_transform(img)
    predictions: List[Tuple[int, float]] = []

    if len(faces) == 0:
        st.error("No face detected, select another image")
        st.image(np.array(img))
        return

    st.success(f" {len(faces)} faces detected, processing image")

    draw_image = PIL.Image.fromarray(img)
    draw = Draw(draw_image)

    for i, (face, bbox) in enumerate(zip(faces, bboxes)):
        face = convert_to_tensor(face, dtype=np.float32)
        face = tf.expand_dims(face, axis=0)

        pred = model.net.predict(face)
        # pred_age = np.argmax(pred, axis=1)[0] + 1

        shape = [(bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3])]
        draw.rectangle(shape, fill=None, outline=COLORS[i], width=5)
        df = pd.DataFrame(list(zip((pred[0] * 100), list(range(1, pred[0].shape[0] + 1)))),
                          columns=['Confidence', 'Age'])
        df = df.sort_values('Confidence', ascending=False).iloc[:5, :].reset_index(drop=True)
        predictions.append(df)

    st.image(np.array(draw_image))

    for i, pred in enumerate(predictions):
        row: pd.Series = pred.loc[0]
        st.markdown(
            f'## Age predicted of face outlined in {COLORS[i]} is <span style="color:{COLORS[i]};">**{row.Age} '
            f'years old**</span> with a model confidence of {round(row.Confidence, 2)} %',
            unsafe_allow_html=True)
        st.markdown(f'### Most 5 confident age predictions for {COLORS[i]} face')
        st.dataframe(pred, use_container_width=True)
    return


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
        predict_from_image(image)
