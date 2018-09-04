import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sensingbee",
    version="0.0.2",
    install_requires=[
       'numpy==1.14.3',
       'pandas==0.23.0',
       'matplotlib==2.1.1',
       'geopandas==0.3.0',
       'scipy==1.0.0',
       'sklearn'
    ],
    author="Adelson Araujo Jr",
    author_email="adelsondias@gmail.com",
    description="Spatial interpolation for sensors data",
    long_description_content_type="text/markdown",
    url="https://github.com/adaj/sensingbee",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
