#!/bin/bash

num_clients=$1

for i in $(seq 1 $num_clients); do
  if [ "$i" -eq "1" ]; then
    python e2e_driver.py --setup &
  else
    python e2e_driver.py &
  fi
done
