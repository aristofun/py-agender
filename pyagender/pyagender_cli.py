import os
import cv2
import argparse
from pyagender.pyagender import PyAgender
from pyagender import VERSION
from keras.utils.data_utils import get_file

# 64x64 (RGB, padded) IMDB dataset trained for 28 epochs
pretrained_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/weights.28-3.73.hdf5"
modhash = 'fbe63257a054c1c5466cfd7bf14646d6'
pretrained_model_filename = "weights.28-3.73.hdf5"


def get_args():
    parser = argparse.ArgumentParser(
        description="Detect and get faces age-gender from a picture file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("IMAGE", type=str, nargs='?', default='', help="path to image file (jpg, png, bmp)")
    parser.add_argument("-v", default=False, action='store_true', help="version info")
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    if args.v:
        print(f'{VERSION}, model: {pretrained_model_filename}')
        exit()

    weight_file = get_file(pretrained_model_filename, pretrained_model,
                           cache_subdir="pretrained_models",
                           file_hash=modhash,
                           cache_dir=os.path.dirname(os.path.abspath(__file__)))
    if not args.IMAGE:
        exit(0)

    pyagender = PyAgender(resnet_weights=weight_file)

    # trained model colorspace seems like BGR https://github.com/yu4u/age-gender-estimation/issues/51
    image = cv2.imread(args.IMAGE)
    faces = pyagender.detect_genders_ages(image)

    if len(faces) < 1:
        exit(0)

    for face in faces:
        print(face)

    # ZEU


if __name__ == '__main__':
    main()
