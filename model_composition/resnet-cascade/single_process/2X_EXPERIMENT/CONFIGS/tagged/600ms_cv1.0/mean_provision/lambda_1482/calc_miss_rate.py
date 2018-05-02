import sys
import os 
import json

def calc(path):
    with open(path, "r") as f:
        results_json = json.load(f)

    client_metrics = results_json["client_metrics"][0]

    miss_count = 0
    per_msg_lats = client_metrics["per_message_lats"]
    for msg_id, latency in per_msg_lats.iteritems():
        if latency > .6:
            miss_count += 1

    print(float(miss_count) / len(per_msg_lats))


if __name__ == "__main__":
    path = sys.argv[1]

    calc(path)
