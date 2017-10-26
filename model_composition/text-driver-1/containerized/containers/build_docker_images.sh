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
cd $DIR/../../../container_utils/
# time docker build -t model-comp/theano-rpc -f TheanoRpcDockerfile ./
# time docker build -t model-comp/tf-rpc -f TfRpcDockerfile ./
time docker build -t model-comp/tf-1-4-rpc -f Tf1-4RpcDockerfile ./

cd $DIR
# Build model-specific images
# time docker build -t model-comp/theano-lstm -f TheanoSentimentDockerfile ./
# time docker build -t model-comp/tf-autocomplete -f TfAutoCompleteDockerfile ./
time docker build -t model-comp/nmt -f NmtDockerfile ./
