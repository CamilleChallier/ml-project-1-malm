#!/bin/bash
conda create -n malm python=3.10.8
conda activate malm
pip install -r requirements.txt
pre-commit install
