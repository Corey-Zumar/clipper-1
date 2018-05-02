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
time docker build -t model-comp/py-rpc -f RpcDockerfile ./
time docker build -t model-comp/tf-rpc -f TfRpcDockerfile ./

cd $DIR
# Build model-specific images
time docker build -t model-comp/tf-kernel-svm -f TfKernelSvmDockerfile ./
time docker build -t model-comp/tf-resnet-feats -f TfResNetDockerfile ./
time docker build -t model-comp/tf-log-reg -f TfLogisticRegressionDockerfile ./
time docker build -t model-comp/vgg-feats -f VggFeaturizationDockerfile ./
time docker build -t model-comp/kpca-svm -f VggKpcaSvmDockerfile ./
time docker build -t model-comp/kernel-svm -f VggKernelSvmDockerfile ./
time docker build -t model-comp/elastic-net -f VggElasticNetDockerfile ./
time docker build -t model-comp/inception-feats -f InceptionFeaturizationDockerfile ./
time docker build -t model-comp/lgbm -f LgbmDockerfile ./
