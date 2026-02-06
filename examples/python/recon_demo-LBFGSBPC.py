# Demo of how to use STIR from python to reconstruct some data with LBFGSBPC

# Copyright 2012-06-05 - 2013 Kris Thielemans
# Copyright 2025 University College London

# This file is part of STIR.
#
# SPDX-License-Identifier: Apache-2.0
#
# See STIR/LICENSE.txt for details

import stir
import matplotlib.pyplot as plt
import os
from LBFGSBPC import LBFGSBPC

stir.Verbosity.set(0)

# Switch 'interactive' mode on for plt.
# Without it, the python shell will wait after every plt.show() for you
# to close the window.
try:
    plt.ion()
except Exception:
    print("Enabling interactive-mode for plotting failed. Continuing.")

# go to directory with input files
os.chdir("../recon_demo")

# %% initialise OSEM reconstruction object and objective function
# we will do this here via a .par file
OSEM_recon = stir.OSMAPOSLReconstruction3DFloat("recon_demo_OSEM.par")
poissonobj = OSEM_recon.get_objective_function()

# %% run initial OSEM

# get initial image
OSEM_target = stir.FloatVoxelsOnCartesianGrid.read_from_file("init.hv")
# we will just fill the whole array with 1 here
OSEM_target.fill(1)

s = OSEM_recon.set_up(OSEM_target)
if not s.succeeded():
    raise RuntimeError("set-up failed")

OSEM_recon.reconstruct(OSEM_target)

# %% add prior/penalty and remove subsets

poissonobj.set_num_subsets(1)
penalty = stir.GibbsRelativeDifferencePenalty3DFloat()
penalty.set_penalisation_factor(1)
poissonobj.set_prior_sptr(penalty)

s = poissonobj.set_up(OSEM_target)

# %% Run reconstruction
recon2 = LBFGSBPC(poissonobj, initial=OSEM_target, update_objective_interval=2)
recon2.process(iterations=15)


# %% make some plots
npimage = recon2.get_output().as_array()
plt.figure()
plt.plot(OSEM_target.as_array()[10, 30, :], label="OSEM")
plt.plot(npimage[10, 30, :], label="LBFGSBPC")
plt.legend()

plt.figure()
plt.imshow(npimage[10, :, :])

plt.figure()
plt.plot(recon2.iterations, recon2.loss)

# %% Keep figures open until user closes them
plt.show(block=True)
