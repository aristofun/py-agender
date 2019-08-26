from pyagender.pyagender import *
# from pyagender.wide_resnet import *

# load version data
with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pyagender', 'version.py')) as f:
    exec(f.read())