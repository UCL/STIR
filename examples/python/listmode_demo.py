# Demo of how to use STIR from Python to read list mode data and translate them 
# into projection data (sinograms).

# To run in "normal" Python, you would type the following in the command line
#  execfile('listmode_demo.py')
# In ipython, you can use
#  %run listmode_demo.py

# Author: Markus Jehl
# Copyright 2022 - Positrigo
# This file is part of STIR.
#
# SPDX-License-Identifier: Apache-2.0
#
# See STIR/LICENSE.txt for details

import stir
import stirextra
import matplotlib.pyplot as plt
import os

# Create instance of LmToProjData.
lm_to_projdata = stir.LmToProjData()

# Define output filename - unfortunately it is not currently possible to extract the output 
# straight from the LmToProjData object and we have to take the detour via the file system.
# The prefix will be appended with "_f1g1d0b0.hs" for the frame we're defining.
lm_to_projdata.set_output_filename_prefix("output_projdata")

# Read in the listmode data from file.
listmode_data = stir.ListModeData.read_from_file("../../recon_test_pack/PET_ACQ_small.l.hdr.STIR")
lm_to_projdata.set_input_data(listmode_data)

# Define the time frames - we are just defining one timeframe starting at 0 and ending at 10s.
time_frames = stir.TimeFrameDefinitions()
time_frames.set_num_time_frames(1)
time_frames.set_time_frame(1, 0, 10)
lm_to_projdata.set_time_frame_definitions(time_frames)

# Read template ProjData.
template_projdata = stir.ProjData.read_from_file("../../recon_test_pack/Siemens_mMR_seg2.hs")
lm_to_projdata.set_template_proj_data_info(template_projdata.get_proj_data_info())

# Perform the actual processing.
lm_to_projdata.set_up()
lm_to_projdata.process_data()

# Read in the generated proj data and visualise them (looks a bit like abstract art, 
# as there are very few counts in this example data).
generated_projdata = stir.ProjData.read_from_file("output_projdata_f1g1d0b0.hs")
first_segment = generated_projdata.get_segment_by_sinogram(0)
projdata_np = stirextra.to_numpy(first_segment)
plt.figure()
plt.imshow(projdata_np[projdata_np.shape[0] // 2, :, :])  # plot the middle slice
plt.title("sinogram 10 (starting from 0)")
plt.clim(0, projdata_np.max() * 0.9)
plt.colorbar()
plt.show(block=True)

# Finally, let's keep the example folder clean.
os.remove("output_projdata_f1g1d0b0.hs")
os.remove("output_projdata_f1g1d0b0.s")
