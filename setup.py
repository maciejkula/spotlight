from setuptools import find_packages, setup


# Import version
__builtins__.__SPOTLIGHT_SETUP__ = True
from spotlight import __version__ as version  # NOQA


setup(
    name='spotlight',
    version=version,
    packages=find_packages(),
    install_requires=['torch>=0.4.0'],
    license='MIT',
    classifiers=['Development Status :: 3 - Alpha',
                 'License :: OSI Approved :: MIT License',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence'],
)
