#!/bin/sh
# you need to run the commands in a bash / linux
# if you have zsh run this docker and execute the script in /data docker run -it -v=$PWD:/data centos:6.7 bash
mkdir data
kaggle competitions download -c dogs-vs-cats -p data --force
cd data
unzip train.zip
cd train
mkdir -p ../sample/train/cats
cp cat.?.jpg cat.??.jpg cat.???.jpg ../sample/train/cats/
mkdir -p ../sample/train/dogs
cp dog.?.jpg dog.??.jpg dog.???.jpg ../sample/train/dogs/
mkdir -p ../sample/valid/cats
cp cat.1[0-3]??.jpg ../sample/valid/cats/
mkdir -p ../sample/valid/dogs
cp dog.1[0-3]??.jpg ../sample/valid/dogs/
