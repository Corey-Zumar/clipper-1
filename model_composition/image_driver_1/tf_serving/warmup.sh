#!/bin/bash

numactl -C 12-40,28-56 python e2e_driver.py -n 1 -l 30 -t 50 -rd .05 -w
