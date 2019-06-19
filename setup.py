import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="modeller",
    version="0.0.1",
    author="Torrance Hodgson",
    author_email="torrance.hodgson@postgrad.curtin.edu.au",
    description="Modeller of visibilities.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/torrance/modeller",
    packages=setuptools.find_packages(),
)
