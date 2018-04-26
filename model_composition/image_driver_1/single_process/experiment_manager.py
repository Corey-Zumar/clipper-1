import sys
import os
import argparse
import subprocess as sp

from config_creator import HIERARCHY_SUBDIR_MEAN_PROVISION, HIERARCHY_SUBDIR_PEAK_PROVISION

def run_exps(tagged_dirs):
    for tagged_dir in tagged_dirs:
        mean_path = os.path.join(tagged_dir, HIERARCHY_SUBDIR_MEAN_PROVISION)
        peak_path = os.path.join(tagged_dir, HIERARCHY_SUBDIR_PEAK_PROVISION)

        reps_dirs = [os.path.join(mean_path, fname) for fname in os.listdir(mean_path)]
        reps_dirs += [os.path.join(peak_path, fname) for fname in os.listdir(peak_path)]
        reps_dirs = sorted(reps_dirs)

        cmds = []
        for reps_dir in reps_dirs:
            tagged_config_paths = [os.path.join(reps_dir, fname) for fname in os.listdir(reps_dir) if "tagged" in fname]
            assert len(tagged_config_paths) == 1
            config_path = tagged_config_paths[0]

            cmd = "python zmq_driver.py -tc {tc}".format(tc=config_path)
            cmds.append(cmd)

        idx = len(cmds) - 1
        while idx >= 0:
            curr_cmd = cmds[idx]
            try:
                sp.check_output(curr_cmd, shell=True)
                idx -= 1
            except Exception as e:
                continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Manage experiments')
    parser.add_argument('-p',  '--paths', type=str, nargs='+', help="Paths to the directories containing experiments to run")

    args = parser.parse_args()

    run_exps(args.paths)

