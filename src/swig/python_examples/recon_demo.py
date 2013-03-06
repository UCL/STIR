# Demo of how to use STIR from python to reconstruct some data

# Copyright 2012-06-05 - $Date$ Kris Thielemans

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


# initialise reconstruction object
# we will do this here via a .par file 
recon=stir.OSMAPOSLReconstruction3DFloat('recon_demo_OSEM.par')
# now modify a few settings from in Python for illustration
recon.set_num_subsets(2);
poissonobj=recon.get_objective_function()
poissonobj.set_sensitivity_filename('sens.hv');
poissonobj.set_recompute_sensitivity(True)

# get initial image
target=stir.FloatVoxelsOnCartesianGrid.read_from_file('init.hv');
# we will just fill the whole array with 1 here
target.fill(1)

# run a few iterations and plot intermediate results
# note: the python shell will wait after every pylab.show() for you
# to close the window. Use ipython or so if you don't like that.
s=recon.set_up(target);
if (s==stir.Succeeded(stir.Succeeded.yes)):
    pylab.figure()
    pylab.hold(True)
    for iter in range(1,4):
        print '\n--------------------- Subiteration ', iter
        recon.set_start_subiteration_num(iter)
        recon.set_num_subiterations(iter)
        s=recon.reconstruct(target);
        poissonobj.set_recompute_sensitivity(False)
        npimage=stirextra.to_numpy(target);
        pylab.plot(npimage[10,30,:])
        pylab.show()

    # plot slice of final image
    pylab.figure()
    pylab.imshow(npimage[10,:,:])
    pylab.show()
else:
    print 'Error setting-up reconstruction object'

