#!/bin/bash

directory="../downloads/SVHN"

mkdir -p ../downloads/$directory

wget -q -O $directory/svhnTrain.zip --progress=bar --show-progress -c http://ufldl.stanford.edu/housenumbers/train.tar.gz

wget -q -O $directory/svhnTest.zip --progress=bar --show-progress -c http://ufldl.stanford.edu/housenumbers/test.tar.gz

wget -q -O $directory/svhnExtra.zip --progress=bar --show-progress -c http://ufldl.stanford.edu/housenumbers/extra.tar.gz
