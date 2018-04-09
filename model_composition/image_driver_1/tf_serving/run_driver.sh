#!/bin/bash

slo_millis=$1
process_path=$2

numactl -C 12-40,28-56 python e2e_driver.py -n 1 -l 90 -t 40 -s $slo_millis -p $process_path
