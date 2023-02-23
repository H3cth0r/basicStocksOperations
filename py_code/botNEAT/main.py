
import os

from neatFunctionality import *


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_file_path = os.path.join(local_dir, "config.txt")
    run(config_file_path)