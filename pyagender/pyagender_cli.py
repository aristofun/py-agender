import os
import cv2
import argparse
from pyagender.pyagender import PyAgender
from keras.utils.data_utils import get_file

# 64x64 (RGB, padded) IMDB dataset trained for 28 epochs
pretrained_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/weights.28-3.73.hdf5"
modhash = 'fbe63257a054c1c5466cfd7bf14646d6'


def get_args():
    parser = argparse.ArgumentParser(
        description="Detect and get faces age-gender from a picture file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("IMAGE", type=str, help="path to image file (jpg, png, bmp)")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    weight_file = get_file("weights.28-3.73.hdf5", pretrained_model,
                           cache_subdir="pretrained_models",
                           file_hash=modhash,
                           cache_dir=os.path.dirname(os.path.abspath(__file__)))

    pyagender = PyAgender(resnet_weights=weight_file)

    # trained model colorspace seems like BGR https://github.com/yu4u/age-gender-estimation/issues/51
    image = cv2.imread(args.IMAGE)
    faces = pyagender.detect_genders_ages(image)

    if len(faces) < 1:
        exit(-1)

    for face in faces:
        print(face)

    # ZEU


if __name__ == '__main__':
    main()
