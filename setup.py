from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.readlines()

setup(
    name="HOTELRESERVATION",
    version="0.1",
    author="Annapurna",
    packages = find_packages(),
    install_requires=requirements,
)