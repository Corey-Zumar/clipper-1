#!/usr/bin/env bash

set -e
set -u
set -o pipefail

unset CDPATH
# one-liner from http://stackoverflow.com/a/246128
# Determines absolute path of the directory containing
# the script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

END=5
for ((i=1000;i<10000;i+=2000)); do
	echo $i
	python $DIR/training/kernel_svm_trainer.py $i $DIR/containers/kernel_svm_model_data/kernel_svm_trained.sav
	python $DIR/driver.py -m kernel-svm -c 17 -b 1 2 4 8 10 16 24 32 48 64
done

for ((i=10000;i<=50000;i+=5000)); do
    echo $i
   	python $DIR/training/kernel_svm_trainer.py $i $DIR/containers/kernel_svm_model_data/kernel_svm_trained.sav
	python $DIR/driver.py -m kernel-svm -c 17 -b 1 2 4 8 10 16 24 32 48 64
done