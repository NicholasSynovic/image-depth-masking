#!/bin/bash

directory="../downloads/MOTS"

mkdir -p ../downloads/$directory

wget -q -O $directory/mots.zip --progress=bar --show-progress -c https://motchallenge.net/data/MOTS.zip
