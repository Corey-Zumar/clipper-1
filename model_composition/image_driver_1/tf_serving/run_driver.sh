#!/bin/bash

process_path=$1

echo $process_path



numactl -C 12-40,28-56 python e2e_driver.py -n 1 -l 30 -t 50 -p $process_path
