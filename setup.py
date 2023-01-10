from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='fibresegt',
    version='0.2',
    author='Rui Guo',
    author_email='rui.guo1@kuleuven.be',
    description='Individual fibre segmentation with PyTorch based on U-Net-ID',
    packages=find_packages(),
    install_requires=required
)