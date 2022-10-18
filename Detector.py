from os import environ, pardir
from os.path import join, dirname, abspath
from typing import Tuple, List, Dict, Any

import numpy as np
from PIL.Image import Image
# from PIL.ImageTk import PhotoImage
from cv2 import cvtColor, createCLAHE, split, merge, resize, COLOR_BGR2LAB, COLOR_LAB2BGR, COLOR_BGR2GRAY, \
    CascadeClassifier, warpAffine, getRotationMatrix2D
from cv2.dnn import readNetFromCaffe, blobFromImage
from numpy import arctan, float32 as npfloat32
from tensorflow.keras.models import load_model

EXT_ROOT = join(dirname(abspath(__file__)))
ROOT_PATH = abspath(join(EXT_ROOT, pardir))
HOME_PATH = environ['HOME']

MODEL_FILE = join(EXT_ROOT, 'data', 'res10_300x300_ssd_iter_140000_fp16.caffemodel')
CONFIG_FILE = join(EXT_ROOT, 'data', "deploy.prototxt")


class FaceDetector:
    __slots__ = 'clahe', 'eye_cascade', 'size', 'net', 'detector', 'min_confidence'

    def __init__(self, min_confidence: float = 0.3):
        self.clahe = createCLAHE(2, (3, 3))
        self.eye_cascade = CascadeClassifier(join(EXT_ROOT, 'data', "haarcascade_eye.xml"))
        self.size = (100, 100, 3)
        self.net = load_model(join(EXT_ROOT, 'model', 'model.h5'), compile=False)
        self.detector = readNetFromCaffe(CONFIG_FILE, MODEL_FILE)
        self.min_confidence = min_confidence

    def detect_faces(self, img: np.ndarray | Image) -> List[Dict[str, Any]]:
        frame_height, frame_width, channels = img.shape
        faces_result: List = []

        blob = blobFromImage(img, 1.0, (100, 100), [106, 121, 150], False, False)
        self.detector.setInput(blob)
        detections = self.detector.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.min_confidence:
                x1 = int(detections[0, 0, i, 3] * frame_width)
                y1 = int(detections[0, 0, i, 4] * frame_height)
                x2 = int(detections[0, 0, i, 5] * frame_width)
                y2 = int(detections[0, 0, i, 6] * frame_height)
                f = (x1, y1, x2 - x1, y2 - y1)
                if f[2] > 1 and f[3] > 1:
                    faces_result.append({
                        'roi': f,
                        'type': 'face',
                        'img': img[f[1]:f[1] + f[3], f[0]:f[0] + f[2]],
                        'confidence': confidence
                    })
        return faces_result

    def align_face(self, img: np.ndarray | Image) -> np.ndarray:
        """Aligns the face detected
        Parameters
        ----------
        img : np.ndarray
            Image which contains the face to detect

        Returns
        -------
        np.ndarray
            new image with aligned face
        """

        gray = cvtColor(img, COLOR_BGR2GRAY)

        eyes = self.eye_cascade.detectMultiScale(gray, 1.05, 6, minSize=(30, 30))
        # roi_gray = gray[y:(y + h), x:(x + w)]
        if len(eyes) < 2:
            print("Eyes not detected, skipping image alignement")
            return img
        left_eye, right_eye = FaceDetector.detect_left_and_right_eye(eyes)

        left_eye_x, left_eye_y = FaceDetector.get_eye_coordinates(left_eye)
        right_eye_x, right_eye_y = FaceDetector.get_eye_coordinates(right_eye)

        delta_x = right_eye_x - left_eye_x
        delta_y = right_eye_y - left_eye_y
        angle = arctan(delta_y / delta_x)
        angle = (angle * 180) / np.pi

        # Width and height of the image
        height, width = img.shape[:2]
        # Calculating a center point of the image
        center = (width // 2, height // 2)
        # Defining a matrix M and calling cv2.getRotationMatrix2D method
        rotation_matrix = getRotationMatrix2D(center, angle, 1.0)
        # Applying the rotation to our image using the cv2.warpAffine method
        rotated = warpAffine(img, rotation_matrix, (width, height))
        return rotated

    def images_preprocessing(self, images: List[np.ndarray | Image]) -> Image:
        """Effettua il preprocessing dell'immagine

        Parameters
        ----------
        image : np.ndarray
        L'immagine da preprocessare
        label: int
        Label associata all'immagine
        resize: bool, optional
        Se true effettua il resize dell'immagine
        size: tuple, optional
        Se resize = True, effettua il resize della dimensione specificata

        Returns
        -------
        tf.Tensor
            L'immagine preprocessata sotto forma di tensore
            :param images:
        """
        for i in range(len(images)):
            img = resize(images[i], self.size[:2])

            # applies clahe
            lab = cvtColor(img, COLOR_BGR2LAB)
            lab_planes = list(split(lab))
            lab_planes[0] = self.clahe.apply(lab_planes[0])
            lab = merge(lab_planes)
            img = cvtColor(lab, COLOR_LAB2BGR)

            # converts the image to float and applies normalization
            img = img.astype(npfloat32)
            images[i] = img / 255.0

        return images

    def fit_transform(self, img: np.ndarray | Image) -> Tuple[List[Tuple[int, int, int, int]]]:
        """Effettua la detection della faccia nell'immagine e allinea il volto.

        Parameters
        ----------
        filename : string
            path dell'immagine da caricare


        Returns
        -------
        Tf.Tensor
            Tensore che contine l'immagine preprocessata
            :param img:
        """

        detected_faces = self.detect_faces(img)
        if len(detected_faces) == 0:
            return ([], [])

        bboxes: List = []
        images: List = []

        for faces in detected_faces:
            roi = faces['roi']
            # rect = rectangle(roi[0], roi[1], roi[0] + roi[2], roi[1] + roi[3])
            # face = self.sp(img_array, rect)
            print(faces['img'].shape)
            images.append(self.align_face(faces['img']))
            bboxes.append(roi)
        return self.images_preprocessing(images), bboxes

    @classmethod
    def detect_left_and_right_eye(cls, eyes: List[Tuple[int, int, int, int]]) -> \
            Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:

        if eyes[0][0] < eyes[1][0]:
            left_eye, right_eye = eyes[0], eyes[1]
        else:
            left_eye, right_eye = eyes[1], eyes[0]
        return left_eye, right_eye

    @staticmethod
    def get_eye_coordinates(eye: Tuple[int, int, int, int]) -> Tuple[int, int]:
        eye_center = (round(eye[0] + (eye[2] / 2)), round(eye[1] + (eye[3] / 2)))
        return eye_center[0], eye_center[1]

    def __hash__(self):
        return hash((self.detector, self.size, self.net, self.clahe, self.min_confidence, self.eye_cascade))
