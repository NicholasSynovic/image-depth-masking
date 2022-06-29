#!/bin/bash

directory="../downloads/CIFAR-10"

mkdir -p ../downloads/$directory

wget -q -O $directory/cifar10.zip --progress=bar --show-progress -c https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
