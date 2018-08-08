# Face Age & Gender detection tool
Simple opencv & tensorflow solution to estimate Age and Gender in your 
next project. Command line style or copy-paste to your Python project. 

Based on and forked from: https://github.com/yu4u/age-gender-estimation

Good enough for pet-projects and prototypes, but not ready for production 
real-life applications

**TODO** train better network with higher image resolution

## Dependencies
- Python 3.5, 3.6
- numpy ~> 1.15
- Keras ~> 2.2
- TensorFlow ~> 1.9
- opencv-python ~> 3.4.2+contrib

Installation (venv Python 3 style): 

```commandline
python3 -m venv --system-site-packages .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

Tested on:
- MacOS 10.13 high Sierra without GPU (you're welcome to update & contribute!)

## Console version

```commandline
python3 pyagender_cli.py PATH_TO_IMAGE
```

TODO: 
- add options (like STDIN input, output formatters etc.) for useful commandline 
applications 
- add help output

## Python version

Copy `pyagender.py`, `wide_resnet.py` and `pretrained_models/*` to your code see the
`pyagender.py` source for reference. Don't forget to download pretrained weights!

```python
from pyagender import *

agender = PyAgender() 
# see available options in __init__() src

faces = agender.detect_genders_ages(cv2.imread(MY_IMAGE))
# [
#   {left: 34, top: 11, right: 122, bottom: 232, width:(r-l), height: (b-t), gender: 0.67, age: 23.5},
#   ...
# ]

# Additional options & methods in PyAgender source
                     
``` 

## Tests

```commandline
python3 -m unittest 
```


# Model

These weigts are from https://github.com/yu4u/age-gender-estimation
on first console version run they are cached in **./pretrained_models** folder:

https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/weights.28-3.73.hdf5
