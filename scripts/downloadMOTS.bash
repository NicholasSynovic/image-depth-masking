#!/bin/bash

directory="../downloads/MOTS"

mkdir ../downloads
mkdir $directory

wget -q -O $directory/mots.zip --progress=bar --show-progress -c https://motchallenge.net/data/MOTS.zip

unzip $directory/*.zip -d $directory

rm $directory/*.zip
