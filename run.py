# create file that can be run with arguments from command line

import argparse
import os
import sys

parser = argparse.ArgumentParser(
    prog="OECT Simulator",
    description="Simulate OECTs with reservoir computing",
    epilog="Enjoy the program! :)",
)

# parser.add_argument()
