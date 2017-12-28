#!/usr/bin/env bash

set -e
set -u
set -o pipefail

unset CDPATH
# one-liner from http://stackoverflow.com/a/246128
# Determines absolute path of the directory containing
# the script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

NUM_PROCS=$1

if [ "$NUM_PROCS" -ge "1" ]
	then
		echo '1'
		(export CUDA_VISIBLE_DEVICES=0; numactl -C 0,16,1,17,2,18,3,19 python driver.py -c 0 16 1 17 2 18 3 19 -ng 0 -b 1 -t 8) &
fi

if [ "$NUM_PROCS" -ge "2" ]
	then
		echo '2'
		(export CUDA_VISIBLE_DEVICES=1; numactl -C 4,20,5,21,6,22,7,23 python driver.py -c 4 20 5 21 6 22 7 23 -ng 0 -b 1 -t 8) &
fi

if [ "$NUM_PROCS" -ge "3" ]
	then
		echo '3'
		(export CUDA_VISIBLE_DEVICES=2; numactl -C 8,24,9,25,10,26,11,27 python driver.py -c 8 24 9 25 10 26 11 27 -ng 0 -b 1 -t 8) &
fi

if [ "$NUM_PROCS" -ge "4" ]
	then
		echo '4'
		(export CUDA_VISIBLE_DEVICES=3; numactl -C 12,28,13,29,14,30,15,31 python driver.py -c 12 28 13 29 14 30 15 31 -ng 0 -b 1 -t 8) &
fi