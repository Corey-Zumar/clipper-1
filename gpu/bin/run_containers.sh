#!/usr/bin/env bash

set -e
set -u
set -o pipefail

unset CDPATH
# one-liner from http://stackoverflow.com/a/246128
# Determines absolute path of the directory containing
# the script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd $DIR

tf_slim_path="$DIR/../tf_models/slim"
checkpoint_path="$DIR/../inception_v3.ckpt"

model_name=$1
clipper_ip=$2
num_containers=$3

python run_containers.py $model_name $clipper_ip $tf_slim_path $checkpoint_path $num_containers
