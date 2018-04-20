import sys
import os
import argparse

import bench_utils
import e2e_utils

import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create configurations for SPD experiments')
    parser.add_argument('-cv',  '--cv', type=float, nargs='+', help="The CV for which to generate configurations")
    parser.add_argument('-p',  '--arrival_procs_path', type=str, help="The path to the arrival processes directory")
    parser.add_argument('-s',  '--slo_profile_path', type=str, help="The path to a JSON profile for a fixed batch size corresponding to some SLO")

    args = parser.parse_args()


