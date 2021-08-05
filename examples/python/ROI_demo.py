# Demo of how to use STIR from Python to compute ROI values using two overloaded versions of the same function.
# The first usage take a STIR shape and internally generates the shape into a voxelised shape before performing ROI
# analysis.
# For the second usage, this python script pre-generates the voxelised shape and passes this as argument to STIR.
# The preprocessing of the voxelised shape reduces the time spent on the evalution of multiple images for a given ROI.

# To run in "normal" Python, you would type the following in the command line
#  execfile('projector_demo.py')
# In ipython, you can use
#  %run projector_demo.py

# Author: Robert Twyman
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
import time
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
print("Creating a EllipsoidalCylinder object as the Region Of Interest (ROI)...")
ROI_shape = stir.EllipsoidalCylinder(10, 5, 4, stir.FloatCartesianCoordinate3D(0, 0, 0))

# number_of_sample needs to be a 3D Int CartesianCoordinate
n = 10
print(f"Setting the number of samples of each axis to be {n}...")
number_of_samples = stir.IntCartesianCoordinate3D(n, n, n)


print("\n\nComputing ROI values using different functions to demonstrate acceleration.")
# Do ROI evaluation
print("Method 1: Pass ROI shape to compute_total_ROI_values...")
print(" - Computing ROI values...")
t0 = time.time()
ROI_eval1 = stir.compute_total_ROI_values(image, ROI_shape, number_of_samples)
t1 = time.time()
print(f" - Time taken = {round(t1-t0,6)} seconds")

# Print ROI details
print("\n - ROI details:")
print(ROI_eval1.report())


print("\nMethod 2: construct the ROI_image for analysis.")
print(" - Constructing ROI volume...")
ROI_volume = image.get_empty_copy()
# construct_volume constructs the ROI_shape in ROI_volume
ROI_shape.construct_volume(ROI_volume, number_of_samples)

print(" - Computing ROI values...")
t2 = time.time()
ROI_eval2 = stir.compute_total_ROI_values(image, ROI_volume)
t3 = time.time()
print(f" - Time taken = {round(t3-t2, 6)} seconds")
# Print ROI details
print("\n - ROI details:")
print(ROI_eval2.report())


# ROI analysis of multiple images should precompute ROI
# However, for a single image the preprocessing in python is likely slower...
print(f"The ROI analysis is ~{round((t1-t0)/(t3-t2),1)}x faster when a ROI volume is passed (Method 2),"
      f" instead of a shape (Method 1).")


