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
    return this->crystal_repeater_z;
}

int
InputStreamFromROOTFileForECATPET::
get_num_dets_per_ring() const
{
    return this->block_repeater * this->crystal_repeater_y;
}


int
InputStreamFromROOTFileForECATPET::
get_num_axial_blocks_per_bucket_v() const
{
    return 1;
}

int
InputStreamFromROOTFileForECATPET::
get_num_transaxial_blocks_per_bucket_v() const
{
    return this->block_repeater;
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

END_NAMESPACE_STIR
