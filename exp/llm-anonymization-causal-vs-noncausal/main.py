import argparse
import sys
from src.utils.initialization import (
    read_config_from_yaml,
    seed_everything,
    set_credentials,
    get_out_file,
)
from src.configs import *
from src.reddit.reddit import run_reddit
from src.anonymized.anonymized import (
    run_anonymized,
    run_eval_inference,
    run_utility_scoring,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/acs_config.yaml",
        help="Path to the config file",
    )
    args = parser.parse_args()

    cfg = read_config_from_yaml(args.config_path)
    seed_everything(cfg.seed)
    set_credentials(cfg)

    f, path = get_out_file(cfg)

    try:
        print(cfg)
        if cfg.task == Task.REDDIT:
            run_reddit(cfg)
        elif cfg.task == Task.ANONYMIZED:
            if cfg.task_config.run_eval_inference:
                run_eval_inference(cfg)
            elif cfg.task_config.run_utility_scoring:
                run_utility_scoring(cfg)
            else:
                run_anonymized(cfg)
        else:
            raise NotImplementedError(f"Task {cfg.task} not implemented")

    except ValueError as e:
        sys.stderr.write(f"Error: {e}")
    finally:
        if cfg.store:
            f.close()
