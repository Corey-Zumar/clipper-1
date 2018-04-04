#!/bin/bash

process_path=$1

echo $process_path

numactl -C 12,28,13,29,14,30,15,31 python e2e_driver.py -n 1 -l 30 -t 50 -p $process_path
