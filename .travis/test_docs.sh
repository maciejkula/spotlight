#!/bin/bash

set -e

cd docs
PATH="$HOME/miniconda/bin:$PATH" make doctest
