import os
import random

import cv2
from pyagender import PyAgender
import numpy as np
from unittest import TestCase

# test repeatability
np.random.seed(2 ** 5)
random.seed(2 ** 5)


def rel(file_path):
    return os.path.join(os.path.dirname(__file__), file_path)


class TestPyAgender(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agender = PyAgender()

    def test_detect_genders_ages(self):
        faces = self.agender.detect_genders_ages(cv2.imread(rel('assets/friends.jpg')))
        self.assertEqual(5, len(faces))  # Rachel not detected!
        # age,gender tuples sorted by age
        ages_genders = [(int(x['age']), 'F' if x['gender'] > 0.5 else 'M') for x in faces]
        # Original detection order: Ross, Chandler, Monica, Joe, Pheobe
        self.assertCountEqual(ages_genders, [(30, 'M'), (29, 'M'), (30, 'F'), (30, 'M'), (29, 'F')])

    def test_gender_age_adultfemales(self):
        w1 = self.agender.detect_genders_ages(cv2.imread(rel('assets/woman38.jpg')))
        self.assertTrue(w1[0]['gender'] > 0.5)
        self.assertEqual(int(w1[0]['age']), 38)

        w2 = self.agender.detect_genders_ages(cv2.imread(rel('assets/woman60.jpg')))
        self.assertTrue(w2[0]['gender'] > 0.5)
        self.assertTrue(int(w2[0]['age']) in range(59, 61))

    def test_gender_age_adultmales(self):
        man44 = self.agender.detect_genders_ages(cv2.imread(rel('assets/man41.jpg')))
        self.assertTrue(man44[0]['gender'] < 0.5)
        self.assertEqual(int(man44[0]['age']), 41)

    def test_gender_age_youngmales(self):
        man1 = self.agender.detect_genders_ages(cv2.imread(rel('assets/man21.jpg')))
        self.assertTrue(man1[0]['gender'] < 0.5)
        self.assertEqual(int(man1[0]['age']), 21)

        man2 = self.agender.detect_genders_ages(cv2.imread(rel('assets/man24.jpg')))
        self.assertTrue(man2[0]['gender'] < 0.5)
        self.assertEqual(int(man2[0]['age']), 24)

    def test_gender_age_youngfemales(self):
        girl1 = self.agender.detect_genders_ages(cv2.imread(rel('assets/gal28.jpg')))
        self.assertTrue(girl1[0]['gender'] > 0.5)
        self.assertEqual(int(girl1[0]['age']), 28)

        g2 = self.agender.detect_genders_ages(cv2.imread(rel('assets/girl2x.jpg')))
        self.assertTrue(g2[0]['gender'] > 0.5)
        self.assertEqual(int(g2[0]['age']), 24)

        sasha = self.agender.detect_genders_ages(cv2.imread(rel('assets/sasha2x.jpg')))
        self.assertTrue(sasha[0]['gender'] > 0.5)
        self.assertEqual(int(sasha[0]['age']), 25)

    def test_aspect_resize(self):
        image = cv2.imread(os.path.join(os.path.dirname(__file__), 'assets/gal28.jpg'))
        new_im = PyAgender.aspect_resize(image, 221, 117)
        self.assertEqual(new_im.shape[:2], (117, 221))
        self.assertEqual(image.shape[:2], (520, 400))

    def test_detect_Xfaces(self):
        faces = self.agender.detect_faces(cv2.imread(rel('assets/friends.jpg')), margin=0.4)
        self.assertEqual(5, len(faces))  # Rachel not detected!
        self.assertCountEqual(
            faces,
            [{'left': 634, 'top': 139, 'right': 848, 'bottom': 353, 'width': 214, 'height': 214},
             # Ross
             {'left': 395, 'top': 213, 'right': 602, 'bottom': 420, 'width': 207, 'height': 207},
             # Chandler
             {'left': 254, 'top': 343, 'right': 445, 'bottom': 534, 'width': 191, 'height': 191},
             # Monica
             {'left': 795, 'top': 349, 'right': 1005, 'bottom': 559, 'width': 210, 'height': 210},
             # Joe
             {'left': 684, 'top': 377, 'right': 897, 'bottom': 590, 'width': 213, 'height': 213}
             # Phoebe
             ])

    def test_detect_1face(self):
        faces = self.agender.detect_faces(cv2.imread(rel('assets/gal28.jpg')), margin=0)
        self.assertEqual(1, len(faces))
        self.assertEqual(
            faces,
            [{'left': 25, 'top': 169, 'right': 354, 'bottom': 498, 'width': 329, 'height': 329}])
