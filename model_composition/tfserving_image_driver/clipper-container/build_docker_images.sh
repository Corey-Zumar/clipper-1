#!/usr/bin/env bash

set -e
set -u
set -o pipefail

unset CDPATH
# one-liner from http://stackoverflow.com/a/246128
# Determines absolute path of the directory containing
# the script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Build RPC base images for python/anaconda and deep learning
# models
cd $DIR/../../container_utils/
time docker build -t model-comp/tf-rpc -f TfRpcDockerfile ./

cd $DIR
# Build model-specific images
time docker build -t model-comp/inceptionv3 -f InceptionV3Dockerfile ./
