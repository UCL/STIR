# Demo to plot the profile of sinograms using STIR (non-TOF)
# To run in "normal" Python, you would type the following in the command line
#  execfile('plot_sinogram_profiles.py')
# In ipython, you can use
#  %run plot_sinogram_profiles.py
# Or of course
#  import plot_sinogram_profiles
# Copyright 2021 University College London

# Authors: Robert Twyman

# This file is part of STIR.
# SPDX-License-Identifier: Apache-2.0
# See STIR/LICENSE.txt for details

import stir
import stirextra
import numpy as np
import argparse
import sys
import matplotlib.pyplot as plt

def plot_sinogram_profiles(projection_data_list, sumaxis=0, select=0):
    """
    Plot a profile through STIR.ProjData.
    Average over all sinograms to reduce noise
    projection_data_list: list of sinogram file names or stir.ProjData objects to load and plot
    sumaxis: axes to sum over (passed to numpy.sum(..., axis))
    select: element to select after summing
    """
    plt.figure()
    ax = plt.subplot(111)

    for f in projection_data_list:
        if isinstance(f, str):
            print(f"Handling:\n  {f}")
            f = stir.ProjData_read_from_file(f)
            label = f

        if isinstance(f, stir.ProjData):
            f = stirextra.to_numpy(f)
            label = ""
            
        else:
            raise ValueError("wrong type of argument")

        plt.plot(np.sum(f, axis=sumaxis)[select,:], label=label)

    ax.legend()
    plt.show()

def main():
    parser = argparse.ArgumentParser(sys.argv[0])
    parser.description= "This script loads, sums axis' and plots profiles over input sinogram files."
    parser.add_argument('filenames', nargs='*',
                        help='Sinogram file names to show, any number.')
    parser.add_argument("--sumaxis", default=0, type=int, 
                        help="Sum all elements on axis (0,1 or 2).")
    parser.add_argument("--select", default=0, type=int,
                        help="Element to show.")
    args = parser.parse_args()
    
    if len(args.filenames) < 1:
        parser.print_help()
        exit(0)

    plot_sinogram_profiles(args.filenames, args.sumaxis, args.select)

if __name__ == '__main__':
    main()
