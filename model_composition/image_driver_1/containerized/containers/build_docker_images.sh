#!/usr/bin/env bash

set -e
set -u
set -o pipefail

unset CDPATH
# one-liner from http://stackoverflow.com/a/246128
# Determines absolute path of the directory containing
# the script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

gcpprefix="gcr.io/clipper-model-comp"
tag="bench"

# Build RPC base images for python/anaconda and deep learning
# models
cd $DIR/../../../container_utils/
# time docker build -t model-comp/py-rpc -f RpcDockerfile ./
time docker build -t model-comp/tf-rpc -f TfRpcDockerfile ./
time docker build -t model-comp/tf-rpc-nogpu -f TfRpcNoGpuDockerfile ./
# time docker build -t model-comp/cuda-rpc -f CudaPyRpcDockerfile ./

cd $DIR
# Build model-specific images
time docker build -t model-comp/tf-kernel-svm -f TfKernelSvmDockerfile ./
docker tag model-comp/tf-kernel-svm $gcpprefix/tf-kernel-svm:$tag
# gcloud docker -- push $gcpprefix/tf-kernel-svm:$tag

time docker build -t model-comp/tf-alexnet-feats -f TfAlexNetDockerfile ./
docker tag model-comp/tf-alexnet-feats $gcpprefix/tf-alexnet-feats:$tag
# gcloud docker -- push $gcpprefix/tf-alexnet-feats:$tag

# time docker build -t model-comp/tf-resnet-feats -f TfResNetDockerfile ./
# docker tag model-comp/tf-resnet-feats $gcpprefix/tf-resnet-feats:$tag
# gcloud docker -- push $gcpprefix/tf-resnet-feats:$tag

# time docker build -t model-comp/tf-log-reg -f TfLogisticRegressionDockerfile ./
# docker tag model-comp/tf-log-reg $gcpprefix/tf-log-reg:$tag
# gcloud docker -- push $gcpprefix/tf-log-reg:$tag

# time docker build -t model-comp/inception-feats -f InceptionFeaturizationDockerfile ./
# docker tag model-comp/inception-feats $gcpprefix/inception-feats:$tag
# gcloud docker -- push $gcpprefix/inception-feats:$tag
