#!/usr/bin/env bash

set -e
set -u
set -o pipefail

unset CDPATH
# one-liner from http://stackoverflow.com/a/246128
# Determines absolute path of the directory containing
# the script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Let the user start this script from anywhere in the filesystem.
cd $DIR/..

# Build tensorflow gpu docker image with libzmq installed
time nvidia-docker build -t tf_zmq_gpu -f ./TfGpuZmqDockerfile ./

# Build RPC container on top of the previous image
time nvidia-docker build -t clipper/py-rpc-gpu -f ./RpcGpuDockerfile ./

# Build tensorflow inception container on top of the rpc+gpu image
time nvidia-docker build -t clipper/tf_gpu_inception_container -f ./TfGpuInceptionDockerfile ./
