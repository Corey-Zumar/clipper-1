#!/bin/bash

numactl -C 12,28,13,29,14,30,15,31 python e2e_driver.py -n 1 -rd .090 -l 30 -t 50

# num_clients=$1
#
# for i in $(seq 1 $num_clients); do
#   if [ "$i" -eq "1" ]; then
#     python e2e_driver.py --setup &
#   else
#     python e2e_driver.py &
#   fi
# done
