/*
    Copyright (C) 2016, UCL
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

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
get_num_axial_crystals_per_block_v() const
{
    return this->crystal_repeater_z;
}

int
InputStreamFromROOTFileForECATPET::
get_num_transaxial_crystals_per_block_v() const
{
    return this->crystal_repeater_y;
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
InputStreamFromROOTFileForECATPET::set_crystal_repeater_x(int val)
{
    crystal_repeater_x = val;
}

void
InputStreamFromROOTFileForECATPET::set_crystal_repeater_y(int val)
{
    crystal_repeater_y = val;
}

void
InputStreamFromROOTFileForECATPET::set_crystal_repeater_z(int val)
{
    crystal_repeater_z = val;
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
