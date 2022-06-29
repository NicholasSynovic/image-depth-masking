#!/bin/bash

directory="../downloads/CTMC_v1"

mkdir -p ../downloads/$directory

wget -q -O $directory/ctmcV1.zip --progress=bar --show-progress -c https://motchallenge.net/data/CTMCV1.zip
