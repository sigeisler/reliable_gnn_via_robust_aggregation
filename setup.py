import logging
import subprocess

from setuptools import setup

import torch

cuda_v = f"cu{torch.version.cuda.replace('.', '')}"
torch_v = torch.__version__.split('.')
torch_v = '.'.join(torch_v[:-1] + ['0'])


def system(command: str):
    output = subprocess.check_output(command, shell=True)
    logging.info(output)


system(f'pip install scipy')
system(f'pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-{torch_v}+{cuda_v}.html')
system(f'pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-{torch_v}+{cuda_v}.html')
system(f'pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-{torch_v}+{cuda_v}.html')
system(f'pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-{torch_v}+{cuda_v}.html')

install_requires = [
    'filelock',
    'numba',
    'pandas',
    'sacred',
    'scikit-learn',
    'scipy',
    'seaborn',
    'seml',
    'tabulate',
    'tinydb',
    'tinydb-serialization',
    'tqdm',
    'torch-geometric'
]

setup(
    name='rgnn',
    version='1.0.0',
    description='Reliable Graph Neural Networks via Robust Aggregation / Message Passing',
    author='Simon Geisler, Daniel Zügner, Stephan Günnemann',
    author_email='geisler@in.tum.de',
    packages=['rgnn'],
    install_requires=install_requires,
    zip_safe=False,
    package_data={'rgnn': ['kernels/csrc/custom.cpp', 'kernels/csrc/custom_kernel.cu']},
    include_package_data=True
)
