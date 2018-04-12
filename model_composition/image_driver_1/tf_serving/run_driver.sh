#!/bin/bash

slo_millis=$1
process_path=$2

numactl -C 11,27,12,28,13,29,14,30,0,16,5,21,6,22 python e2e_driver.py -n 1 -l 60 -t 30 -s $slo_millis -p $process_path
