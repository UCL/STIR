/*
    Copyright (C) 2016, UCL
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

START_NAMESPACE_STIR

int
InputStreamFromROOTFileForECATPET::
get_num_rings() const
{
   return this->block_repeater_z * this->crystal_repeater_z;
}

int
InputStreamFromROOTFileForECATPET::
get_num_dets_per_ring() const
{
    return this->block_repeater_y * this->crystal_repeater_y;
}


int
InputStreamFromROOTFileForECATPET::
get_num_axial_blocks_per_bucket_v() const
{
    return this->block_repeater_z;
}

int
InputStreamFromROOTFileForECATPET::
get_num_transaxial_blocks_per_bucket_v() const
{
    return this->block_repeater_y;
}

int
InputStreamFromROOTFileForECATPET::
get_num_axial_crystals_per_singles_unit() const
{
    return this->crystal_repeater_z;
}

int
InputStreamFromROOTFileForECATPET::
get_num_trans_crystals_per_singles_unit() const
{
    return this->crystal_repeater_y;
}

void
InputStreamFromROOTFileForECATPET::set_block_repeater_y(int val)
{
    block_repeater_y = val;
}

void
InputStreamFromROOTFileForECATPET::set_block_repeater_z(int val)
{
    block_repeater_z = val;
}

END_NAMESPACE_STIR
