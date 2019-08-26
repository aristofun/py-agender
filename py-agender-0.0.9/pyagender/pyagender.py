import os
import sys
import cv2
import numpy as np


sys.setrecursionlimit(2 ** 20)
# np.random.seed(2 ** 10)

from pyagender.wide_resnet import WideResNet


class PyAgender():
    """
    Age-Gender estimator with built-in OpenCV face detector.

    Based on https://github.com/asmith26/wide_resnets_keras ResNet
    and pre trained model from https://github.com/yu4u/age-gender-estimation
    """

    def __init__(self,
                 haar_cascade='pretrained_models/haarcascade_frontalface_default.xml',
                 cv_scaleFactor=1.1,
                 cv_minNeighbors=8,
                 cv_minSize=(64, 64),
                 cv_flags=cv2.CASCADE_SCALE_IMAGE,
                 resnet_weights='pretrained_models/weights.28-3.73.hdf5',
                 resnet_imagesize=64):
        self.cv_face_detector = cv2.CascadeClassifier(
            os.path.join(os.path.dirname(__file__), haar_cascade))
        self.cv_scaleFactor = cv_scaleFactor
        self.cv_minNeighbors = cv_minNeighbors
        self.cv_minSize = cv_minSize
        self.cv_flags = cv_flags
        self.resnet_imagesize = resnet_imagesize
        self.resnet = WideResNet(image_size=resnet_imagesize)()
        self.resnet.load_weights(os.path.join(os.path.dirname(__file__), resnet_weights))

    def detect_genders_ages(self, image):
        """
        Detects all faces using OpenCV Haar cascades options provided in the constructor

        :param image: image to detect
        :return: array of dicts or empty array (no detections)
                 {left: 34, top: 11, right: 122, bottom: 232, width:(r-l), height: (b-t),
                  gender: 0.67, age: 23.5}
                  
                 gender > 0.5 == female
        """
        faceregions = self.detect_faces(image, margin=0.4)

        for face in faceregions:
            face['gender'], face['age'] = self.gender_age(image,
                                                          left=face['left'], top=face['top'],
                                                          width=face['width'],
                                                          height=face['height'])

        return faceregions

    def gender_age(self, face_image, left=0, top=0, width=None, height=None):
        """
        Assuming ready to use face region on input (no detection to do)
        
        :param face_image: CV image object to feed into CNN age-gender estimator
        :param left,top left upper corner of a face in the image
        :param w,h width,height of a face (face_image == the whole face by default)

        :return: [gender, age] evaluation (gender > 0.5 == female, age is float)
        """
        img_h, img_w, __ = np.shape(face_image)

        if width is not None:
            img_w = width
        if height is not None:
            img_h = height

        # Crop & resize image to 64pix box
        test_img = PyAgender.aspect_resize(face_image[top:top + img_h, left:left + img_w],
                                           self.resnet_imagesize, self.resnet_imagesize)

        # predict ages and genders of the detected faces
        result = self.resnet.predict(np.array([test_img]))
        predicted_genders = result[0]
        ages = np.arange(0, 101).reshape(101, 1)
        predicted_ages = result[1].dot(ages).flatten()
        # print(f'gender: {predicted_genders}')
        # print(f'predicted_ages: {predicted_ages}')

        return predicted_genders[0][0], predicted_ages[0]

    def detect_faces(self, image, margin=0.2):
        """
        :param image: Original image (in opencv BGR) to find faces on
        :param padding: additional margin widht/height percentage
        :return:
            array of face image rectangles {left: 34, top: 11, right: 122, bottom: 232, width:(r-l), height: (b-t)}
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_h, img_w = np.shape(gray)

        face_results = []

        faces = self.cv_face_detector.detectMultiScale(gray,
                                                       scaleFactor=self.cv_scaleFactor,
                                                       minNeighbors=self.cv_minNeighbors,
                                                       minSize=self.cv_minSize,
                                                       flags=self.cv_flags)
        for (x, y, w, h) in faces:
            xi1 = max(int(x - margin * w), 0)
            xi2 = min(int(x + w + margin * w), img_w - 1)
            yi1 = max(int(y - margin * h), 0)
            yi2 = min(int(y + h + margin * h), img_h - 1)
            detection = {'left': xi1, 'top': yi1, 'right': xi2, 'bottom': yi2,
                         'width': (xi2 - xi1), 'height': (yi2 - yi1)}
            face_results.append(detection)

        return face_results

    @staticmethod
    def aspect_resize(image, width, height, padding=cv2.BORDER_REPLICATE, color=[0, 0, 0]):
        """
        Letterboxing image resize preserving original aspect ratio, padding if necessary
        :param image: cv2 compatible (x,y,3) shape image data
        :param width: desired width in pixels
        :param height: desired height in pixels
        :param padding: OpenCV padding strategy (see https://docs.opencv.org/3.1.0/d3/df2/tutorial_py_basic_ops.html)
        :param color: padding cv2 color for cv2.BORDER_CONSTANT padding strategy

        :return: resized copy of image in cv2 default format
        """
        old_size = image.shape[:2]
        # determine the longest side
        ratio = min(float(height) / old_size[0], float(width) / old_size[1])
        # resize accordingly
        new_size = tuple([int(x * ratio) for x in old_size])
        im = cv2.resize(image, (new_size[1], new_size[0]))

        delta_w = width - new_size[1]
        delta_h = height - new_size[0]

        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        new_im = cv2.copyMakeBorder(im, top, bottom, left, right, padding, value=color)
        return new_im
