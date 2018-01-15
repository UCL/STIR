# Demo of how to use STIR from Python to project some data
# To run in "normal" Python, you would type the following in the command line
#  execfile('projector_demo.py')
# In ipython, you can use
#  %run projector_demo.py

# Copyright 2014-2015, 2017 - University College London
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

#%% Initial imports
import stir
import stirextra
import pylab
import numpy
import os
#%% go to directory with input files
os.chdir('../recon_demo')
#%% Read it some projection data
projdata=stir.ProjData.read_from_file('smalllong.hs');
#%% Create an empty image with suitable voxel sizes
# use smaller voxels than the default
zoom=2.;
target=stir.FloatVoxelsOnCartesianGrid(projdata.get_proj_data_info(), zoom);
#%% initialise the projection matrix 
# Using ray-tracing here
# Note that the default is to restrict the projection to a cylindrical FOV
projmatrix=stir.ProjMatrixByBinUsingRayTracing();
projmatrix.set_up(projdata.get_proj_data_info(), target);
#%% construct projectors
forwardprojector=stir.ForwardProjectorByBinUsingProjMatrixByBin(projmatrix);
backprojector=stir.BackProjectorByBinUsingProjMatrixByBin(projmatrix);

#%% create projection data for output of forward projection
# We'll just create the data in memory here
projdataout=stir.ProjDataInMemory(projdata.get_exam_info(), projdata.get_proj_data_info());
# Note: we could write to file, but it is right now a bit complicated to open a
# file for read/write:
#  inout=stir.ios.trunc|stir.ios.ios_base_in|stir.ios.out;
#  projdataout=stir.ProjDataInterfile(projdata.get_exam_info(), projdata.get_proj_data_info(), 'my_test_python_projection.hs',inout);
#%% forward project an image. Here just some uniform data
target.fill(2);
forwardprojector.forward_project(projdataout, target);
#%% display
seg=projdataout.get_segment_by_sinogram(0);
seg_array=stirextra.to_numpy(seg);
pylab.figure();
pylab.subplot(1,2,1)
pylab.imshow(seg_array[10,:,:]);
pylab.title('Forward projection')
pylab.subplot(1,2,2)
pylab.plot(seg_array[10,0,:])
pylab.show()
#%% backproject this projection data
# we need to set the target to zero first, otherwise it will add to existing numbers.
target.fill(0)
backprojector.back_project(target, projdataout);
#%% display
# This shows a beautiful pattern, a well-known feature of a ray-tracing matrix
target_array=stirextra.to_numpy(target);
pylab.figure();
pylab.subplot(1,2,1)
pylab.imshow(target_array[10,:,:]);
pylab.title('Back-projection')
pylab.subplot(1,2,2)
pylab.plot(target_array[10,80,:])
pylab.show()
#%% Let's use more LORs per sinogram bin (which will be a bit slower of course)
projmatrix.set_num_tangential_LORs(10);
# Need to call set_up again
projmatrix.set_up(projdata.get_proj_data_info(), target);
#%% You could re-run the forward projection, but we'll skip that for now
# forwardprojector.forward_project(projdataout, target);
#%% Run another backprojection and display
target.fill(0)
backprojector.back_project(target, projdataout);
new_target_array=stirextra.to_numpy(target);
pylab.figure();
pylab.subplot(1,2,1)
pylab.imshow(new_target_array[10,:,:]);
pylab.title('Back-projection with 10 LORs per bin')
pylab.subplot(1,2,2)
pylab.plot(new_target_array[10,80,:])
pylab.show()
#%% compare profiles to check if overall features are fine
pylab.figure()
pylab.plot(target_array[10,80,:])
pylab.hold('on')
pylab.plot(new_target_array[10,80,:])
pylab.title('comparing both profiles')
pylab.show()
