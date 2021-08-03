# Demo of how to use STIR from Python to compute ROI values
# To run in "normal" Python, you would type the following in the command line
#  execfile('projector_demo.py')
# In ipython, you can use
#  %run projector_demo.py

# Copyright 2021 - University College London
# This file is part of STIR.
#
# SPDX-License-Identifier: Apache-2.0
#
# See STIR/LICENSE.txt for details

#%% Initial imports
import stir
import stirextra
import os
import numpy as np
#%% go to directory with input files

print("\nRunning ROI_demo.py")
os.chdir('../recon_demo')
image_filename = 'init.hv'

# Load image from file (for shape)
print(f"Loading {image_filename} as image...")
image = stir.FloatVoxelsOnCartesianGrid.read_from_file(image_filename)

print("Populating image with random values...")
# Make a copy in numpy, populate random voxel value and back-fill STIR FloatVoxelsOnCartesianGrid object
image_numpy = stirextra.to_numpy(image)
image_numpy = np.random.random(image_numpy.shape)
image.fill(image_numpy.flat)

# Construct an example shape
print("Constructing a EllipsoidalCylinder as the Region Of Interest (ROI)...")
ROI_shape = stir.EllipsoidalCylinder(10, 5, 4, stir.FloatCartesianCoordinate3D(0, 0, 0))



# number_of_sample needs to be a 3D Int CartesianCoordinate
n = 10
print(f"Setting the number of samples of each axis to be {n}...")
number_of_samples = stir.IntCartesianCoordinate3D(n, n, n)

# Do ROI evaluation
print("Computing ROI value...")
ROI_eval = stir.compute_total_ROI_values(image, ROI_shape, number_of_samples)

# Print the mean and standard deviation
print(f"ROI mean value = {ROI_eval.get_mean()}")
print(f"ROI stddev value = {ROI_eval.get_stddev()}")

