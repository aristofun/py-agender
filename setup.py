import setuptools

REQUIRED_PACKAGES = [
    'numpy >= 1.12',
    'Keras >= 2.1',
    'TensorFlow >= 1.10, < 1.19',
    'opencv-python >= 3.3.0+contrib, < 3.99'
]

CONSOLE_SCRIPTS = [
    'py-agender = pyagender.pyagender_cli:main'
]

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="py-agender",
    version="0.0.3",
    author="Michael Butlitsky",
    author_email="aristofun@yandex.ru",
    description="Simple opencv & tensorflow based solution to estimate Faces, Age and Gender on pictures",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aristofun/py-agender",
    packages=setuptools.find_packages(),
    classifiers=(
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        "Environment :: Console",
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        'License :: Other/Proprietary License',
        "Operating System :: OS Independent",
    ),
    zip_safe=False,
    install_requires=REQUIRED_PACKAGES,
    entry_points={'console_scripts': CONSOLE_SCRIPTS})
