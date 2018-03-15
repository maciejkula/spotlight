#!/bin/bash

set -e

export PATH="$HOME/miniconda/bin:$PATH"
source activate conda_env

pip install -e .
