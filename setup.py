import pyagender
import setuptools

with open('requirements.txt') as f:
    REQUIRED_PACKAGES = f.read().split('\n')

EXTRAS_REQUIRE = {
    "cpu": ["tensorflow>=1.10.0, < 1.19.0"],
    "gpu": ["tensorflow-gpu>=1.10.0, < 1.19.0"],
}

CONSOLE_SCRIPTS = [
    'py-agender = pyagender.pyagender_cli:main'
]

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="py-agender",
    version=pyagender.VERSION,
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
    include_package_data=True,
    zip_safe=False,
    install_requires=REQUIRED_PACKAGES,
    extras_require=EXTRAS_REQUIRE,
    entry_points={'console_scripts': CONSOLE_SCRIPTS}
)
