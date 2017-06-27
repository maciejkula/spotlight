from setuptools import setup


setup(
    name='spotlight',
    version='0.1.0',
    install_requires=['numpy',
                      'scipy',
                      'h5py',
                      'requests'],
    packages=['spotlight'],
    license='MIT',
    classifiers=['Development Status :: 3 - Alpha',
                 'License :: OSI Approved :: MIT License',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence'],
)
