#!/bin/bash

directory="../downloads/HT_21"

mkdir -p ../downloads/$directory

wget -q -O $directory/ht21.zip --progress=bar --show-progress -c https://motchallenge.net/data/HT21.zip
