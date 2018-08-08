import os
import cv2
from pyagender import PyAgender
from unittest import TestCase


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
        ages_genders = sorted([(int(x['age']), 'F' if x['gender'] > 0.5 else 'M') for x in faces],
                              key=lambda x: x[0])
        # Ross, Chandler, Monica, Phoebe, Joe
        self.assertEqual(ages_genders, [(25, 'M'), (28, 'M'), (29, 'F'), (29, 'F'), (31, 'M')])

    def test_gender_age_adultfemales(self):
        w40 = self.agender.detect_genders_ages(cv2.imread(rel('assets/woman40.jpg')))
        self.assertTrue(w40[0]['gender'] > 0.5)
        self.assertEqual(int(w40[0]['age']), 40)

        w40 = self.agender.detect_genders_ages(cv2.imread(rel('assets/woman61.jpg')))
        self.assertTrue(w40[0]['gender'] > 0.5)
        self.assertEqual(int(w40[0]['age']), 61)

    def test_gender_age_adultmales(self):
        man44 = self.agender.detect_genders_ages(cv2.imread(rel('assets/man41.jpg')))
        self.assertTrue(man44[0]['gender'] < 0.5)
        self.assertEqual(int(man44[0]['age']), 41)

    def test_gender_age_youngmales(self):
        man21 = self.agender.detect_genders_ages(cv2.imread(rel('assets/man21.jpg')))
        self.assertTrue(man21[0]['gender'] < 0.5)
        self.assertEqual(int(man21[0]['age']), 21)

        man25 = self.agender.detect_genders_ages(cv2.imread(rel('assets/man25.jpg')))
        self.assertTrue(man25[0]['gender'] < 0.5)
        self.assertEqual(int(man25[0]['age']), 25)

    def test_gender_age_youngfemales(self):
        g27 = self.agender.detect_genders_ages(cv2.imread(rel('assets/gal27.jpg')))
        self.assertTrue(g27[0]['gender'] > 0.5)
        self.assertEqual(int(g27[0]['age']), 27)

        g23 = self.agender.detect_genders_ages(cv2.imread(rel('assets/girl23.jpg')))
        self.assertTrue(g23[0]['gender'] > 0.5)
        self.assertEqual(int(g23[0]['age']), 23)

        sasha = self.agender.detect_genders_ages(cv2.imread(rel('assets/sasha24.jpg')))
        self.assertTrue(sasha[0]['gender'] > 0.5)
        self.assertEqual(int(sasha[0]['age']), 24)

    def test_aspect_resize(self):
        image = cv2.imread(os.path.join(os.path.dirname(__file__), 'assets/gal27.jpg'))
        new_im = PyAgender.aspect_resize(image, 221, 117)
        self.assertEqual(new_im.shape[:2], (117, 221))
        self.assertEqual(image.shape[:2], (520, 400))

    def test_detect_Xfaces(self):
        faces = self.agender.detect_faces(cv2.imread(rel('assets/friends.jpg')))
        self.assertEqual(5, len(faces))  # Rachel not detected!
        self.assertCountEqual(
            faces,
            [{'left': 658, 'top': 163, 'right': 824, 'bottom': 329, 'width': 166, 'height': 166},
             # Ross
             {'left': 418, 'top': 236, 'right': 579, 'bottom': 397, 'width': 161, 'height': 161},
             # Chandler
             {'left': 275, 'top': 364, 'right': 424, 'bottom': 513, 'width': 149, 'height': 149},
             # Monica
             {'left': 818, 'top': 372, 'right': 982, 'bottom': 536, 'width': 164, 'height': 164},
             # Joe
             {'left': 708, 'top': 401, 'right': 873, 'bottom': 566, 'width': 165, 'height': 165},
             # Phoebe
             ])

    def test_detect_1face(self):
        faces = self.agender.detect_faces(cv2.imread(rel('assets/gal27.jpg')), margin=0)
        self.assertEqual(1, len(faces))
        self.assertEqual(
            faces,
            [{'left': 25, 'top': 169, 'right': 354, 'bottom': 498, 'width': 329, 'height': 329}])
