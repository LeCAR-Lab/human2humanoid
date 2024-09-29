from setuptools import find_packages
from distutils.core import setup

setup(
    name='legged_gym',
    version='1.0.0',
    author='',
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email='',
    description='Isaac Gym environments for Humanoid Robots (More Specically, unitree H1)',
    install_requires=['isaacgym',
                      'rsl-rl',
                      'matplotlib',
                      'rich',
                      'wandb',
                      'termcolor',
                      'tensorboard']
)