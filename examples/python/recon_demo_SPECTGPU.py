# Demo of how to use STIR from python to reconstruct some data
# To run in "normal" Python, you would type the following in the command line
#  execfile('recon_demo.py')
# In ipython, you can use
#  %run recon_demo.py

# Copyright 2026 Daniel Deidda
# Copyright 2026 National Physical Laboratory

# This file is part of STIR.
#
# SPDX-License-Identifier: Apache-2.0
#
# See STIR/LICENSE.txt for details

from matplotlib import pylab

import stir
import matplotlib.pyplot as plt
from pathlib import Path
import os

script_dir = Path(__file__).resolve().parent

os.chdir(script_dir / ".." / "recon_demo")

# initialise reconstruction object
# we will do this here via a .par file
recon = stir.OSMAPOSLReconstruction3DFloat('recon_demo_OSEM_SPECTGPU.par')
# now modify a few settings from in Python for illustration
recon.set_num_subsets(2)
num_subiterations = 4
# set filenames to save subset sensitivities (for illustration purposes)
poissonobj = recon.get_objective_function()
poissonobj.set_subsensitivity_filenames('sens_subset%d.hv')
poissonobj.set_recompute_sensitivity(True)

# get initial image
target = recon.get_initial_data()#stir.FloatVoxelsOnCartesianGrid.read_from_file('init.hv')
# we will just fill the whole array with 1 here
target.fill(1)

# run a few iterations and plot intermediate results

# Switch 'interactive' mode on for pylab.
# Without it, the python shell will wait after every pylab.show() for you
# to close the window.
try:
    plt.ion()
except:
    print("Enabling interactive-mode for plotting failed. Continuing.")

s = recon.set_up(target)
if (s.succeeded()):
    fig,  ax = plt.subplots()
    for i in range(1,num_subiterations+1):
        print('\n--------------------- Subiteration ', i)
        recon.set_start_subiteration_num(i)
        recon.set_num_subiterations(i)
        s = recon.reconstruct(target)
        # currently we need to explicitly prevent recomputing sensitivity
        # when we call reconstruct() again in the next iteration
        poissonobj.set_recompute_sensitivity(False)
        # extract to python for plotting
        npimage = target.as_array()
        slice_idx = npimage.shape[0] // 2
        row_idx = npimage.shape[1] // 2
        ax.clear()
        ax.plot(npimage[slice_idx, row_idx, :])
        ax.set_title(f"Subiteration {i}")
        ax.set_xlabel("Voxel")
        ax.set_ylabel("Value")

        plt.draw()
        plt.pause(0.01)

    # Save the final image
    recon_filename = f"recon_image_{num_subiterations}.hv"
    print(f"Saving Image as: {recon_filename}")
    target.write_to_file(recon_filename)

    # display a reconstructed slice
    

    fig2, ax2 = plt.subplots()

    im = ax2.imshow(npimage[slice_idx, :, :],
                    cmap="gray",
                    origin="lower")

    ax2.axhline(row_idx,
                color="red",
                linewidth=2,
                label=f"Profile row {row_idx}")

    ax2.set_title(f"Slice {slice_idx}")
    ax2.legend()

    fig2.colorbar(im, ax=ax2)
    

    plt.show(block=True)
else:
    print('Error setting-up reconstruction object')

