#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate xai
pip install git+https://github.com/Kojoley/atari-py.git