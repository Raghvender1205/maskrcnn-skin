from setuptools import setup, find_packages

setup(
    name="maskrcnn",
    version="0.0.1",
    author="Raghvender",
    author_email="raghvender1205@gmail.com",
    description="MaskRCNN implemented on TensorFlow 2.x",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Raghvender1205/maskrcnn-skin.git",
    packages=find_packages(),
    install_requires=[
        "numpy",
        'opencv-python',
        "tensorflow",
        "matplotlib",
        "pillow"
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',  # Change as appropriate
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.6',
)
