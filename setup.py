import sys

if sys.version_info < (3, 6):
    sys.exit('megatron requires Python >= 3.6')

from setuptools import setup, find_packages
from pathlib import Path
setup(
    name='megatron',
    version='0.1a',
    author='Huidong Chen',
    athor_email='huidong.chen AT mgh DOT harvard DOT edu',
    license='MIT',
    description='MEGA TRajectories of clONes',
    long_description=Path('README.md').read_text('utf-8'),
    long_description_content_type="text/markdown",
    url='https://github.com/pinellolab/MEGATRON',
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
