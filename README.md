# Face Age & Gender detection tool
Simple opencv & tensorflow solution to estimate Age and Gender in your 
next project. Command line or Python DIY style. 

Based on and forked from: https://github.com/yu4u/age-gender-estimation

Good enough for pet-projects and prototypes, but not ready for production 
real-life applications

## How

```commandline
pip3 install py-agender
```
Warning — ~190MB download (pretrained network is heavy).

CLI: 

```commandline
py-agender PATH_TO_IMAGE
```

Python:

```python
from pyagender import PyAgender

agender = PyAgender() 
# see available options in __init__() src

faces = agender.detect_genders_ages(cv2.imread(MY_IMAGE))
# [
#   {left: 34, top: 11, right: 122, bottom: 232, width:(r-l), height: (b-t), gender: 0.67, age: 23.5},
#   ...
# ]

# Additional options & methods in PyAgender source
``` 

Don't forget to download pretrained weights if using source code DIY style.

## TODO: 
- add options (like STDIN input, output formatters etc.) for useful commandline 
applications 
- add help output
- train better network with higher image resolution

## Tests

```commandline
python3 -m unittest 
```

## Dependencies
- Python 3.5, 3.6
- numpy ~> 1.15
- Keras ~> 2.2
- TensorFlow ~> 1.9
- opencv-python ~> 3.4.2+contrib

Tested on:
- MacOS 10.13 high Sierra without GPU (you're welcome to update & contribute!)


# Model

These weigts are from https://github.com/yu4u/age-gender-estimation
on first console version run they are cached in **./pyagender.pretrained_models** folder:

https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/weights.28-3.73.hdf5

# License

This project is released under the MIT license.

However, [the IMDB-Wiki dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) used for pretrained network above is originally provided under the following conditions:

> Please notice that this dataset is made available for academic research purpose only. All the images are collected from the Internet, and the copyright belongs to the original owners. If any of the images belongs to you and you would like it removed, please kindly inform us, we will remove it from our dataset immediately.

Refer to fresh IMDB-Wiki dataset License and act accordingly.