#!/bin/bash

# secure shell
set -e -u  


echo "Start demo"
source setupenv/bin/activate
python -m jupyter notebook train.ipynb
