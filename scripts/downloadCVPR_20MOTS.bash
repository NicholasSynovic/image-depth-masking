#!/bin/bash

directory="../downloads/CVPR_20MOTS"

mkdir ../downloads
mkdir $directory

wget -q -O $directory/cvpr20MOTS.zip --progress=bar --show-progress -c https://motchallenge.net/data/CVPRMOTS20.zip
