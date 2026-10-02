
    Copyright (C) 2008- 2012, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

This directory contains a set of utilities and scripts to make it easier to
use STIR together with SimSET.

WARNING: SimSET has miriad options. The files here assume that you run SimSET in 
a certain way. There are hardly any checks if this was the case or not.

WARNING: An important caveat is that STIR can at presently not handle SimSET data 
with an even number of tangential positions ('num_td_bins') with a symmetric range 
for min_td and max_td. This is because STIR uses a convention suitable for ECAT data 
taking interleaving of sinogram-bins into account. You will get artifacts in the 
images unless you use an odd number for num_td_bins.
Similarly, usually in SimSET you put the centre of the image in the centre of the 
scanner (although you don't have to of course). The current version of STIR will 
only do this if you have an odd number of pixels in x and y.

Most scripts require bash. Some require python. We also use various standard 
utilities such as awk, grep, tr.

Available utilities:
(see code for more info)

- conv_to_SimSET_att_image
Allows converting an image with attenuation factor for 511 keV photons to a (8-bit)
index file that can be used as input for SimSET.

- conv_SimSET_projdata_to_STIR.sh
Allows converting SimSET output sinograms ("weight files" to STIR Interfile files.
(Do not use the executable of the same name (without .sh) unless in emergencies).
Read the header of the script for some information.

- SimSET_STIR_names.sh
Prints the (root of) the names of the projection data constructed by the above script.

- make_hv_from_Simset_params.sh
Construct an Interfile header for a binary (attenuation or emission) image
output by SimSET.

- write_phg_image_info
Helps constructing a PHG input file by generating the object-spec.
You would not normally run this directly, but use stir_image_to_simset_object.sh

- stir_image_to_simset_object.sh
Helps constructing a PHG input file by generating the object-spec for an image
(as interpreted by list_image_info).
You probably don't need this if you use run_SimSET.sh

- run_SimSET.sh
A script to run SimSET. It takes input images that STIR can read, and a few templates
input files.

- add_SimSET_results.sh
A script to add results from run_SimSET.sh from different simulations. Useful if for
example to increase statistics.
WARNING: this script first calls mult_num_photons.sh in an attempt to handle cases where
you have different number of decays per simulations. However, this most definitely does 
not handle cases where you change importance sampling parameters.

- conv_SimSET_crystal_pairs_to_STIR.sh
Converts SimSET weight files to STIR Interfile projection data for simulations that
model block detectors (detector_type = block in det.rec) with the output binned by
crystal pair (bin_by_crystal = true in bin.rec). The weight file then holds one
value per pair of crystals instead of SimSET's usual sinogram bins (num_td_bins,
num_aa_bins, ...), so conv_SimSET_projdata_to_STIR.sh cannot be used for it.
Usage:

    conv_SimSET_crystal_pairs_to_STIR.sh output_prefix template.hs weight_file

  output_prefix  outputs are written as <output_prefix>_<label>.hs (and .s)
  template.hs    STIR projection data header describing the simulated scanner and
                 the output sinogram (segments, views, tangential positions)
  weight_file    the weight_image_path of the bin.rec file

The script reads the SimSET binning from the weight file header (this needs
SIMSET_DIR, see "How to use" below) and supports:
  scatter_param 4, min_s 0, max_s 1 : 2 grids [unscattered][scattered], written as
                                      <output_prefix>_unscattered and
                                      <output_prefix>_singles_scat
  scatter_param 5, min_s 2, max_s 2 : 1 grid [multiple scatter], written as
                                      <output_prefix>_multiples
The block layout is taken from the template, so the template has to have the same
block layout as the simulated detector. The script then calls the executable
conv_SimSET_crystal_pairs_to_STIR. Example:

    conv_SimSET_crystal_pairs_to_STIR.sh sino template.hs rec.weight

- conv_SimSET_crystal_pairs_to_STIR
The executable called by the script above. You only need to run it directly if
your binning is not one of the two above, or to give the SimSET block layout by
hand. Usage:

    conv_SimSET_crystal_pairs_to_STIR output_prefix template.hs weight_file mode \
        num_block_rings num_blocks_per_ring \
        crystals_per_block_axial crystals_per_block_transaxial

  output_prefix, template.hs, weight_file  as for the script
  mode           multiples    : the weight file holds 1 grid (e.g. scatter_param 5,
                                min_s = max_s = 2), written as
                                <output_prefix>_multiples
                 singles_scat : the weight file holds 2 grids (e.g. scatter_param 4,
                                min_s 0, max_s 1), written as
                                <output_prefix>_unscattered and
                                <output_prefix>_singles_scat
  The number of grids in the weight file has to match the mode.

The scanner is taken from the template header ("Number of rings",
"Number of detectors per ring", "Number of crystals per block in axial/transaxial
direction"). The last four arguments describe the simulated SimSET detector, and
are read by hand from the SimSET detector files:
  num_block_rings                number of block rings in det.rec that contain
                                 active crystals (rings with only shielding do
                                 not count)
  num_blocks_per_ring            ring_num_blocks_in_ring in the ring parameter file
  crystals_per_block_axial       number of active crystals per block along z
  crystals_per_block_transaxial  number of active crystals per block along y
(when every element of the block is an active crystal, the last two are
block_layer_num_z_changes + 1 and block_layer_num_y_changes + 1).
The utility stops if these do not agree with the template: the crystals per block
have to be the same, num_block_rings x crystals_per_block_axial has to equal the
number of rings, and num_blocks_per_ring x crystals_per_block_transaxial the number
of detectors per ring. The total number of crystals is also checked against the
size of the weight file.

How the conversion works:
A weight file is a 32768-byte SimSET header followed by one or more grids of
num_crystals x num_crystals floats, one value per crystal pair. SimSET numbers
the active crystals block by block (block rings in the order of det.rec, blocks
in the order of the ring file, counter-clockwise), and inside a block along y
first, then z. For a STIR detector (ring, det) this gives
    block_index = (ring / crystals_per_block_axial) * num_blocks_per_ring
                  + det / crystals_per_block_transaxial
    crystal     = block_index * crystals_per_block_axial * crystals_per_block_transaxial
                  + (ring % crystals_per_block_axial) * crystals_per_block_transaxial
                  + det % crystals_per_block_transaxial
For each bin of the output sinogram, STIR's get_all_det_pos_pairs_for_bin()
gives the detector pairs that fall in that bin (more than one if the template
uses axial compression or view mashing). The grid values of the matching
crystal pairs are added together to give the bin value.
SimSET stores each coincidence once, in the upper triangle of the grid
(crystal1 < crystal2), so each pair is ordered before the look-up. Anything on or
below the diagonal is reported and ignored.
The views are stored in reverse order (view -> num_views - 1 - view). SimSET's y
axis points up and STIR's points down, so an image written row by row for SimSET
is the same picture in both, but the two rings are numbered in opposite
directions round the object. Reversing the views undoes this mirror. The exact
alignment depends on the "View offset" in the template.

Each grid is read into memory in turn (4 x num_crystals^2 bytes, e.g. 5.8 GB for
38016 crystals). Only Cylindrical and BlocksOnCylindrical, non-arc-corrected,
non-TOF templates are supported. Example, for 6 block rings of 22 blocks of
12 x 24 (axial x transaxial) crystals:

    conv_SimSET_crystal_pairs_to_STIR sino template.hs rec.weight singles_scat 6 22 12 24

How to use
-----------
You first have to tell these routines where your SimSET installation is located, for instance

    SIMSET_DIR=~/simset/2.9.1
    export SIMSET_DIR

All STIR utilities/scripts have to be in your path, e.g. if your 
INSTALL_PREFIX was ~/STIR-bin:

    PATH=$PATH:~/STIR-bin/bin

See the SimSET/examples directory for an example.
