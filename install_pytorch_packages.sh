#!/bin/bash
# This file is for debug use only.

# Use 'cpu' or a cuda version e.g. 'cu102'.
CUDA=cu110
TORCH=1.7.0
echo "Installing Pytorch, Pytorch Geometric and its dependencies."
pip install --upgrade -f https://download.pytorch.org/whl/cu110/torch_stable.html torch==${TORCH}+${CUDA}
pip install --upgrade torch-optimizer
pip install --upgrade torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install --upgrade torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install --upgrade torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install --upgrade torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install --upgrade torch-geometric
