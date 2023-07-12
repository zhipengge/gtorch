#!/bin/bash

conda create -n gtorch python=3.8.10 -y
source activate gtorch
pip install -r requirements.txt