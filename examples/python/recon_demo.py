# Demo of how to use STIR from python to reconstruct some data
# To run in "normal" Python, you would type the following in the command line
#  execfile('recon_demo.py')
# In ipython, you can use
#  %run recon_demo.py

# Copyright 2012-06-05 - 2013 Kris Thielemans
# Copyright 2015 University College London

# This file is part of STIR.
#
# This file is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or
# (at your option) any later version.
#
# This file is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# See STIR/LICENSE.txt for details

import stir
import stirextra
import pylab
import numpy
import os

# go to directory with input files
os.chdir('../recon_demo')

# initialise reconstruction object
# we will do this here via a .par file 
recon=stir.OSMAPOSLReconstruction3DFloat('recon_demo_OSEM.par')
# now modify a few settings from in Python for illustration
recon.set_num_subsets(2);
# set filenames to save subset sensitivities (for illustration purposes)
poissonobj=recon.get_objective_function()
poissonobj.set_subsensitivity_filenames('sens_subset%d.hv');
poissonobj.set_recompute_sensitivity(True)

# get initial image
target=stir.FloatVoxelsOnCartesianGrid.read_from_file('init.hv');
# we will just fill the whole array with 1 here
target.fill(1)

# run a few iterations and plot intermediate results

# Switch 'interactive' mode on for pylab.
# Without it, the python shell will wait after every pylab.show() for you
# to close the window. 
pylab.ion()

s=recon.set_up(target);
if (s==stir.Succeeded(stir.Succeeded.yes)):
    pylab.figure()
    pylab.hold(True)
    for iter in range(1,4):
        print('\n--------------------- Subiteration ', iter)
        recon.set_start_subiteration_num(iter)
        recon.set_num_subiterations(iter)
        s=recon.reconstruct(target);
        # currently we need to explicitly prevent recomputing sensitivity when we
        # call reconstruct() again in the next iteration
        poissonobj.set_recompute_sensitivity(False)
        # extract to python for plotting
        npimage=stirextra.to_numpy(target);
        pylab.plot(npimage[10,30,:])
        pylab.show()

    # plot slice of final image
    pylab.figure()
    pylab.imshow(npimage[10,:,:])
    pylab.show()
else:
    print('Error setting-up reconstruction object')

