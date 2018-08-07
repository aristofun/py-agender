import os

import cv2
from wide_resnet import WideResNet


class PyAgender():
    """
    Age-Gender estimator with built-in OpenCV face detector.

    Based on https://github.com/asmith26/wide_resnets_keras ResNet
    and pre trained model from https://github.com/yu4u/age-gender-estimation
    """

    def __init__(self,
                 haar_cascade='/pretrained_models/haarcascade_frontalface_default.xml',
                 cv_scaleFactor=1.2,
                 cv_minNeighbors=7,
                 cv_minSize=(64, 64),
                 cv_flags=cv2.CASCADE_SCALE_IMAGE,
                 resnet_weights='/pretrained_models/weights.28-3.73.hdf5',
                 resnet_imagesize=64
                 ):
        self.cv_face_detector = cv2.CascadeClassifier(
            os.path.join(os.path.dirname(__file__), haar_cascade))
        self.cv_scaleFactor = cv_scaleFactor
        self.cv_minNeighbors = cv_minNeighbors
        self.cv_minSize = cv_minSize
        self.cv_flags = cv_flags
        self.resnet_imagesize = resnet_imagesize
        self.resnet = WideResNet(image_size=resnet_imagesize)()
        self.resnet.load_weights(os.path.join(os.path.dirname(__file__), resnet_weights))
        #

    def detect_genders_ages(self, image):
        """
        Detects all faces using OpenCV Haar cascades options provided in the constructor

        :param image: image to detect
        :return: array of dicts or empty array (no detections)
                 each dict == {x: 34, y: 11, w:67, h: 68, gender: 0.67, age: 23.5}
        """

    def gender_age(self, face_image, x=0, y=0, w=None, h=None):
        """
        Assuming ready to use face on input (no detection to do)
        :param face_image: CV image object to feed into CNN age-gender estimator
        :param x,y left upper corner of a face in the image
        :param w,h width,height of a face (face_image == the whole face by default)

        :return: [gender, age] evaluation (gender > 0.5 == female, age is float)
        """
