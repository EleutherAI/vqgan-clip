from setuptools import setup, find_packages

setup(
    name='vqgan-clip',
    version='0.0.1',
    description='VQGAN-CLIP: Zero-Shot Semantic Image Generation and Editing',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
    ],
)
