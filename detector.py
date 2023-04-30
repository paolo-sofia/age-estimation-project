from os.path import abspath, dirname, join
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import PIL.Image
import tensorflow as tf

EXT_ROOT: str = join(dirname(abspath(__file__)))
MODEL_FILE: str = join(EXT_ROOT, 'model', 'res10_300x300_ssd_iter_140000_fp16.caffemodel')
CONFIG_FILE: str = join(EXT_ROOT, 'model', "deploy.prototxt")


class FaceDetector:
    __slots__ = 'clahe', 'eye_cascade', 'size', 'net', 'detector', 'min_confidence'

    def __init__(self, min_confidence: float = 0.3):
        self.clahe: cv2.CLAHE = cv2.createCLAHE(2, (3, 3))
        self.eye_cascade: cv2.CascadeClassifier = cv2.CascadeClassifier(join(EXT_ROOT, 'model', "haarcascade_eye.xml"))
        self.size: Tuple[int, int, int] = (100, 100, 3)
        self.net: tf.keras.engine.functional.Functional = tf.keras.models.load_model(
            join(EXT_ROOT, 'model', 'model.h5'), compile=False)
        self.detector: cv2.dnn.Net = cv2.dnn.readNetFromCaffe(CONFIG_FILE, MODEL_FILE)
        self.min_confidence: float = min_confidence

    def detect_faces(self, img: np.ndarray | PIL.Image.Image) -> List[Dict[str, Any]]:
        """
        Detect all the faces in the image using caffe face detection model
        :param img: the image in which to detect faces
        :return: a List containing detected faces info. Each element is a dictionary containing data about the detected
        face (roi, type, original_image, confidence)
        """
        frame_height, frame_width, channels = img.shape
        faces_result: List = []

        blob: np.ndarray = cv2.dnn.blobFromImage(img, 1.0, (100, 100), [106, 121, 150], False, False)

        self.detector.setInput(blob)
        detections: np.ndarray = self.detector.forward()

        for i in range(detections.shape[2]):
            confidence: float = detections[0, 0, i, 2]
            if confidence > self.min_confidence:
                x1: int = int(detections[0, 0, i, 3] * frame_width)
                y1: int = int(detections[0, 0, i, 4] * frame_height)
                x2: int = int(detections[0, 0, i, 5] * frame_width)
                y2: int = int(detections[0, 0, i, 6] * frame_height)
                f: Tuple[int, int, int, int] = (x1, y1, x2 - x1, y2 - y1)
                if f[2] > 1 and f[3] > 1:
                    faces_result.append({
                        'roi': f,
                        'type': 'face',
                        'img': img[f[1]:f[1] + f[3], f[0]:f[0] + f[2]],
                        'confidence': confidence
                    })
        return faces_result

    def align_face(self, img: np.ndarray | PIL.Image.Image) -> np.ndarray:
        """
        Aligns the detected face to feed to age estimation model
        :param img: the detected face
        :return: the aligned face
        """
        gray: np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        eyes: np.ndarray | Tuple = self.eye_cascade.detectMultiScale(gray, 1.05, 6, minSize=(30, 30))
        # roi_gray = gray[y:(y + h), x:(x + w)]

        if not isinstance(eyes, np.ndarray):
            return None

        if eyes.shape[0] < 2:
            print("Eyes not detected, skipping image alignement")
            return img
        left_eye, right_eye = FaceDetector.detect_left_and_right_eye(eyes)

        left_eye_x, left_eye_y = FaceDetector.get_eye_coordinates(left_eye)
        right_eye_x, right_eye_y = FaceDetector.get_eye_coordinates(right_eye)

        delta_x: int = right_eye_x - left_eye_x
        delta_y: int = right_eye_y - left_eye_y
        angle: float = (np.arctan(delta_y / delta_x) * 180) / np.pi

        # Width and height of the image
        height, width = img.shape[:2]
        # Calculating a center point of the image
        center: Tuple[int, int] = (width // 2, height // 2)
        # Defining a matrix M and calling cv2.getRotationMatrix2D method
        rotation_matrix: np.ndarray = cv2.getRotationMatrix2D(center, angle, 1.0)
        # Applying the rotation to our image using the cv2.warpAffine method
        rotated: np.ndarray = cv2.warpAffine(img, rotation_matrix, (width, height))
        return rotated

    def images_preprocessing(self, images: List[np.ndarray | PIL.Image.Image]) -> List[np.ndarray | PIL.Image.Image]:
        """
        Preprocess the image to be feed to the model. It resizes the image, applies clahe and normalize the tensor
        :param images: the images to preprocess
        :return: the preprocessed images
        """
        for i, image in enumerate(images):
            image: np.ndarray = cv2.resize(image, self.size[:2])

            # applies clahe
            lab: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab_planes: List[np.ndarray] = list(cv2.split(lab))
            lab_planes[0] = self.clahe.apply(lab_planes[0])
            lab = cv2.merge(lab_planes)
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            # converts the image to float and applies normalization
            image = image.astype(np.float32)
            images[i] = image / 255.0

        return images

    def fit_transform(self, img: np.ndarray | PIL.Image.Image) \
            -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]:
        """
        Preprocess the image by detecting all the faces in the image and aligns them to be feed to the age estimator
        :param img: the image to preprocess
        :return: A tuple of 2 elements, the first one contains the preprocessed images, the second contains the list of
        bounding boxes of detected faces
        """
        detected_faces: List[Dict[str, Any]] = self.detect_faces(img)
        if len(detected_faces) == 0:
            return [], []

        bboxes: List = []
        images: List = []

        for faces in detected_faces:
            roi: Tuple[int, int, int, int] = faces['roi']
            # rect = rectangle(roi[0], roi[1], roi[0] + roi[2], roi[1] + roi[3])
            # face = self.sp(img_array, rect)
            aligned_face: Optional[np.ndarray] = self.align_face(faces['img'])
            if aligned_face is not None:
                images.append(aligned_face)
                bboxes.append(roi)
        return self.images_preprocessing(images), bboxes

    @classmethod
    def detect_left_and_right_eye(cls, eyes: List[Tuple[int, int, int, int]]) -> \
            Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
        """
        Detect which is the left eye and which is the right eye from the eyes list
        :param eyes: a list containing the roi of all the detected eyes
        :return: the detected left eye and the detected right eye
        """
        if eyes[0][0] < eyes[1][0]:
            left_eye, right_eye = eyes[0], eyes[1]
        else:
            left_eye, right_eye = eyes[1], eyes[0]
        return left_eye, right_eye

    @staticmethod
    def get_eye_coordinates(eye: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """
        Computes the coordinate x,y of the given eye
        :param eye: the eye represented as a tuple of coordinates (roi)
        :return: the x, y coordinates of the eye
        """
        return round(eye[0] + (eye[2] / 2)), round(eye[1] + (eye[3] / 2))

    def __hash__(self):
        return hash((self.detector, self.size, self.net, self.clahe, self.min_confidence, self.eye_cascade))
