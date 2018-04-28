import sys
import os

from e2e_utils import load_tagged_arrival_deltas, tag_arrival_process

if __name__ == "__main__":
    process_path = sys.argv[1]
    output_path = sys.argv[2]
    num_replicas = int(sys.argv[3])

    tag_arrival_process(process_path, output_path, num_replicas)
    tagged_process = load_tagged_arrival_deltas(output_path)


