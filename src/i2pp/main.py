"""Main routine of img2physiprop."""

import argparse
import logging
import os

import yaml
from i2pp.core.run import run_i2pp
from munch import munchify


def main() -> None:
    """Call img2physiprop runner with config.

    Raises:
        RuntimeError: If provided config is not a valid file.
    """

    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--config_file_path",
        "-cfp",
        help="Path to config file.",
        type=str,
        default="src/i2pp/main_example_config.yaml",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.config_file_path):
        raise RuntimeError(
            "Config file not found! img2physiprop can not be executed!"
        )

    # load config and convert to simple namespace for easier access
    with open(args.config_file_path, "r") as file:
        config = munchify(yaml.safe_load(file))

    # execute i2pp
    run_i2pp(config)


if __name__ == "__main__":  # pragma: no cover

    main()
    exit(0)
