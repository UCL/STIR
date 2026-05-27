import sys
import argparse

from . import launch_GUI

parser = argparse.ArgumentParser(sys.argv[0])
parser.description = ("Qt5 based visualisation of projection data.")
parser.add_argument('-f', '--filename',
                    type=str,
                    help='Projection data file name to show')
args = parser.parse_args()


launch_GUI(args.filename)

