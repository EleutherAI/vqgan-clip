from setuptools import setup, find_packages

setup(
    name='vqgan_clip',
    version='0.0.1',
    description='VQGAN-CLIP: Zero-Shot Semantic Image Generation and Editing',
    url='https://github.com/eleutherai/vqgan-clip',
    author='Placeholder',
    author_email='contact@eleuther.ai',
    license='MIT',
    packages=find_packages(),
    install_requires=['clip', 'taming-transformers'],
    scripts={},

)
