# Demo of how to use STIR to compute detector coordinates, LORs etc from bins or vice versa.
# This demo is PET specific and only works for "non-arccorrected" data.
#
# We use class methods to compute coordinates and make some plots.
# As an illustration, we also backproject a bin and plot that.
# If all works well, the 2 sets of plots should look similar!
#
# This demo will only make sense if you look at the code... Adapt for your needs!

# Copyright 2026 University College London
# Author Kris Thielemans

# This file is part of STIR.
#
# SPDX-License-Identifier: Apache-2.0
#
# See STIR/LICENSE.txt for details


# %% imports

import matplotlib.pyplot as plt
import numpy as np
import stir
import stirextra
import os

# %% read in example data
# feel free to use your own, although you might have to adjust the SSRB statement below
template_filename = os.path.join(
    stir.get_STIR_examples_dir(), "samples/PET_TOF_Interfile_header_Signa_PETMR.hs"
)
template_projdata = stir.ProjData.read_from_file(template_filename)
org_proj_data_info = template_projdata.get_proj_data_info()
# %% reduce size a bit to speed-up backprojection below
# (and avoid a current STIR bug in ProjDataInMemory when there are more than 2^31 bins in the proj_data.
# see https://github.com/UCL/STIR/issues/1505)
# do help(stir.SSRB) to find out about arguments
proj_data_info = stir.SSRB(org_proj_data_info, 3, 2, 0, -1, 9)
# %% save some variables for shorter code below
scanner = proj_data_info.get_scanner()
rmax = scanner.get_inner_ring_radius()
scanner_len = scanner.get_ring_spacing() * scanner.get_num_rings()
num_rings = scanner.get_num_rings()
num_dets_per_ring = scanner.get_num_detectors_per_ring()

# %% print some information
print(proj_data_info.parameter_info())
print(
    scanner.get_num_detectors_per_ring(),
    proj_data_info.get_num_views(),
    proj_data_info.get_num_tangential_poss(),
    proj_data_info.get_min_segment_num(),
    proj_data_info.get_max_segment_num(),
    proj_data_info.get_num_segments(),
)
seg_num = proj_data_info.get_max_segment_num()
print(
    proj_data_info.get_min_ring_difference(seg_num),
    proj_data_info.get_num_axial_poss(seg_num),
)

# %% Define some example detection positions, find the corresponding bin, coordinates etc
dp1 = stir.DetectionPosition(
    5, num_rings - 1
)  # first arg number along ring, 2nd arg: ring
dp2 = stir.DetectionPosition(
    num_dets_per_ring // 2,
    dp1.axial_coord
    + proj_data_info.get_min_ring_difference(proj_data_info.get_min_segment_num()),
)
# Use the max TOF bin. However, note that DetectionPositionPair.timing_pos_num is using the scanner's,
# i.e. unmashed, TOF bins.
timing_pos = proj_data_info.get_max_tof_pos_num() * proj_data_info.get_tof_mash_factor()
dp_pair = stir.DetectionPositionPair(dp1, dp2, timing_pos)
bin = stir.Bin()
proj_data_info.get_bin_for_det_pos_pair(bin, dp_pair)
lor_sino = stir.LORInAxialAndNoArcCorrSinogramCoordinates()
proj_data_info.get_LOR(lor_sino, bin)
lor = stir.LORAs2Points(lor_sino)
print(dp_pair)
print(bin)
print(lor.p1(), lor.p2())
# %% make a plot of the current LOR
fig, axs = plt.subplots(1, 2)
ax = axs[0]
ax.set_xlim([-rmax * 1.1, rmax * 1.1])
ax.set_ylim([-rmax * 1.1, rmax * 1.1])
ax.set_xlabel("X ")
ax.set_ylabel("Y ")
# In STIR's coordinate system, the y axis points "downwards", see the STIR-developers-overview.
ax.invert_yaxis()
ax.set_aspect("equal", "box")
ax = axs[1]
ax.set_xlim([-scanner_len / 2, scanner_len / 2])
ax.set_ylim([-rmax * 1.1, rmax * 1.1])
ax.set_xlabel("Z ")
ax.set_ylabel("Y ")
ax.set_aspect("equal", "box")

p1 = lor.p1()
p2 = lor.p2()

fig.suptitle("LOR with a green circle at first detector")
axs[0].plot((p1.x(), p2.x()), (p1.y(), p2.y()))
axs[0].plot(p1.x(), p1.y(), "og")
axs[0].set_title("Transverse")
axs[1].plot((p1.z(), p2.z()), (p1.y(), p2.y()))
axs[1].plot(p1.z(), p1.y(), "og")
axs[1].set_title("Sagital")  # label only ok if HFS
plt.show()

# %%%%%%% Verification with projector

# %% create new proj_data to store results
proj_data_out = stir.ProjDataInMemory(template_projdata.get_exam_info(), proj_data_info)
proj_data_out.fill(0)  # optionally fill sinogram with 0 (when experimenting)
# increment the current bin (incrementing, in case you want to try multiple bins)
# %%
bin.bin_value = proj_data_out.get_bin_value(bin) + 1
proj_data_out.set_bin_value(bin)
# optionally write to file
# proj_data_out.write_to_file('mytest.hs')

# %% Create an empty image with suitable voxel sizes
# use smaller voxels than the default
zoom = 0.5
target = stir.FloatVoxelsOnCartesianGrid(
    proj_data_out.get_exam_info(), proj_data_out.get_proj_data_info(), zoom
)
# Currently need to resize, due to https://github.com/UCL/STIR/issues/1706

target.resize(
    stir.IndexRange3D(
        stir.make_IntCoordinate(0, target.get_min_y(), target.get_min_x()),
        stir.make_IntCoordinate(
            2 * num_rings - 2, target.get_max_y(), target.get_max_x()
        ),
    )
)

first_voxel_coords, last_voxel_coords = (
    stirextra.get_physical_coordinates_for_bounding_box(target)
)
# %% initialise the projection matrix and projectors
# Using ray-tracing here (could use Parallelproj instead)
# Note that the default is to restrict the projection to a cylindrical FOV
projmatrix = stir.ProjMatrixByBinUsingRayTracing()
backprojector = stir.BackProjectorByBinUsingProjMatrixByBin(projmatrix)
backprojector.set_up(proj_data_out.get_proj_data_info(), target)
# %% back-project!
backprojector.start_accumulating_in_new_target()
backprojector.back_project(proj_data_out)
backprojector.get_output(target)
# %% Plot image in the same way as the plot above
# Unfortunately, we currently have a mismatch in coordinate systems in axial direction
# See https://github.com/UCL/STIR/issues/219
# We cope with that by computing the shift as the centre of the image
z_origin_shift = (first_voxel_coords.z() + last_voxel_coords.z()) / 2
np_image = target.as_array()
fig, axs = plt.subplots(1, 2)
ax = axs[0]
ax.set_xlim([-rmax * 1.1, rmax * 1.1])
ax.set_ylim([-rmax * 1.1, rmax * 1.1])
ax.set_xlabel("X ")
ax.set_ylabel("Y ")
ax.set_aspect("equal", "box")
ax = axs[1]
ax.set_xlim([-scanner_len / 2, scanner_len / 2])
ax.set_ylim([-rmax * 1.1, rmax * 1.1])
ax.set_xlabel("Z ")
ax.set_ylabel("Y ")
ax.set_aspect("equal", "box")
fig.suptitle(f"MIPs of Backprojection\n{bin}\ngreen circle at first detector")
axs[0].set_title("Transverse")
axs[1].set_title("Sagital")  # label only ok if HFS
axs[0].imshow(
    np.max(np_image, axis=0),
    extent=[
        first_voxel_coords.x(),
        last_voxel_coords.x(),
        first_voxel_coords.y(),
        last_voxel_coords.y(),
    ],
    origin="lower",
)
axs[1].imshow(
    np.max(np_image, axis=2).transpose(),
    extent=[
        first_voxel_coords.z() - z_origin_shift,
        last_voxel_coords.z() - z_origin_shift,
        first_voxel_coords.y(),
        last_voxel_coords.y(),
    ],
    origin="lower",
)
axs[0].plot(p1.x(), p1.y(), "og")
axs[1].plot(p1.z(), p1.y(), "og")
plt.show()
# %%
# If you used TOF data above, you should see that the max TOF bin selects voxels closest to dp_pair.pos1()
