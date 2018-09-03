import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sensingbee",
    version="0.0.2",
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
