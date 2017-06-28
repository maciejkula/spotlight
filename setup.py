from setuptools import setup


# Import version
__builtins__.__SPOTLIGHT_SETUP__ = True
from spotlight import __version__ as version  # NOQA


setup(
    name='spotlight',
    version=version,
    packages=['spotlight'],
    license='MIT',
    classifiers=['Development Status :: 3 - Alpha',
                 'License :: OSI Approved :: MIT License',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence'],
)
