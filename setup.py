import setuptools

with open("README.md", "r") as fh:
    long_description = fh

setuptools.setup(
    name="pywdf",
    version="0.0.1",
    author="Gustav Anthon",
    author_email="anthon.gus1@gmail.com",
    description="A framework for wave digital filter circuits",
    long_description=long_description,
    url= "https://github.com/gusanthon/pywdf",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 2.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)