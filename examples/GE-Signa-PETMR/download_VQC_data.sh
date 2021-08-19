#! /bin/sh -e
# A simple script to get the VQC data-set from Zenodo
# Author: Kris Thielemans

curl -L https://zenodo.org/record/3887517/files/VQC_Phantom_Dataset.zip?download=1 -o VQC_Phantom_Dataset.zip
unzip VQC_Phantom_Dataset.zip
