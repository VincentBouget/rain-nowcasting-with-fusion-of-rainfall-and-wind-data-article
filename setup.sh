#!/bin/bash

# secure shell
set -e -u  

which >/dev/null python3 || { echo "Error: python3 required"; exit 1; }
which >/dev/null wget || { echo "Error: wget required"; exit 1; }

echo "Install python3 env (setupenv)"
python3 -m venv setupenv
source setupenv/bin/activate
pip install --upgrade pip
sed 's/=/==/;s/pytorch/torch/' <requirements.txt >/tmp/requirements.txt

echo "Install requirements"
pip install -r /tmp/requirements.txt
rm -f /tmp/requirements.txt
pip install jupyter

echo "Download dataset"
wget --no-check-certificate https://www-pequan.lip6.fr/~bereziat/rain-nowcasting/data.tar.gz
tar xvfz data.tar.gz
rm -f data.tar.gz



