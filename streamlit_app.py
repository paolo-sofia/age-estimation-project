import PIL.Image
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL.ImageDraw import Draw
from tensorflow.python.framework.ops import convert_to_tensor

from Detector import FaceDetector

COLORS = ('red', 'green', 'blue', 'yellow', 'cyano', 'orange')


@st.cache(hash_funcs={FaceDetector: hash})
def load_model():
    face_detector = FaceDetector()
    return face_detector


def predict_from_image(img: np.ndarray):
    # -------------------- MODEL PREDICT --------------------
    model = load_model()
    faces, bboxes = model.fit_transform(img)
    predictions = []

    if len(faces) == 0:
        st.error("No face detected, select another image")
        st.image(np.array(img))
    else:
        st.success(f" {len(faces)} faces detected, processing image")

        draw_image = PIL.Image.fromarray(img)
        draw = Draw(draw_image)

        for i, (face, bbox) in enumerate(zip(faces, bboxes)):
            face = convert_to_tensor(face, dtype=np.float32)
            face = tf.expand_dims(face, axis=0)

            predictions.append(np.argmax(model.net.predict(face), axis=1)[0] + 1)

            shape = [(bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3])]
            draw.rectangle(shape, fill=None, outline=COLORS[i], width=5)

        st.image(np.array(draw_image))

        for i, pred in enumerate(predictions):
            st.markdown(
                f'# Age predicted of face outlined in {COLORS[i]} is <span style="color:{COLORS[i]};">**{pred} years old**</span>',
                unsafe_allow_html=True)


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
