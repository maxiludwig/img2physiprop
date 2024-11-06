"""Runner which executes the main routine of img2physiprop."""

import logging
import time
from typing import Any

from i2pp.core.example import exemplary_function
from i2pp.core.utilities import RunManager

log = logging.getLogger("i2pp")


def run_i2pp(config: Any) -> None:
    """General run procedure of img2physiprop.

    Args:
        config (Any): Munch type object containing all configs for current
        run. Config options can be called via attribute-style access.
    """

    # Time
    start_time = time.time()

    # Run manager to handle overall tasks
    run_manager = RunManager(config)
    run_manager.init_run()

    # add overall execution here
    log.info("Output of exemplary program: " + str(exemplary_function(2, 4)))

    # finalize run
    run_manager.finish_run(start_time)
