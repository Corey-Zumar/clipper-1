import sys
import os
import json

LAMBDAS_CONFIG_KEY_SLO = "slo"
LAMBDAS_CONFIG_KEY_COST = "cost"
LAMBDAS_CONFIG_KEY_LAMBDA = "lam"
LAMBDAS_CONFIG_KEY_CV = "cv"

def parse_lambdas(lambdas_dir_path, output_file_path):
    lambdas_files = [os.path.join(lambdas_dir_path, fname) for fname in os.listdir(lambdas_dir_path) if "json" in fname and "swp" not in fname]

    by_slo_lambdas = {}
    for lambdas_file in lambdas_files:
        with open(lambdas_file, "r") as f:
            lambdas_config = json.load(f)
            for config_item in lambdas_config:
                slo = config_item[LAMBDAS_CONFIG_KEY_SLO]
                cost = config_item[LAMBDAS_CONFIG_KEY_COST]
                lambda_val = config_item[LAMBDAS_CONFIG_KEY_LAMBDA]
                cv = config_item[LAMBDAS_CONFIG_KEY_CV]
    
                if slo not in by_slo_lambdas:
                    by_slo_lambdas[slo] = { cv : [lambda_val] }
                else:
                    if cv not in by_slo_lambdas[slo]:
                        by_slo_lambdas[slo][cv] = [lambda_val]
                    elif lambda_val not in by_slo_lambdas[slo][cv]:
                        by_slo_lambdas[slo][cv].append(lambda_val)
                
                print("SLO: {slo}, LAMBDA: {lv}, COST: {cost}".format(slo=slo, lv=lambda_val, cost=cost))

    with open(output_file_path, "w") as f:
        json.dump(by_slo_lambdas, f, indent=4)

    print("Wrote SLO-keyed lambda values to file with path: {}".format(output_file_path))

if __name__ == "__main__":
    lambdas_dir_path = sys.argv[1]
    output_path = sys.argv[2]
    parse_lambdas(lambdas_dir_path, output_path)
