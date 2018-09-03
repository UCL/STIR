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
InputStreamFromROOTFileForCylindricalPET::
get_num_rings() const
{
    return static_cast<int>( this->crystal_repeater_z * this->module_repeater_z *
                             this->submodule_repeater_z);
}

int
InputStreamFromROOTFileForCylindricalPET::
get_num_dets_per_ring() const
{
    return static_cast<int>(this->rsector_repeater * this->module_repeater_y *
                            this->submodule_repeater_y * this->crystal_repeater_y);
}


int
InputStreamFromROOTFileForCylindricalPET::
get_num_axial_blocks_per_bucket_v() const
{
    return this->module_repeater_z;
}

int
InputStreamFromROOTFileForCylindricalPET::
get_num_transaxial_blocks_per_bucket_v() const
{
    return this->module_repeater_y;
}

int
InputStreamFromROOTFileForCylindricalPET::
get_num_axial_crystals_per_block_v() const
{
    return static_cast<int>(this->crystal_repeater_z *
                            this->module_repeater_z);
}

int
InputStreamFromROOTFileForCylindricalPET::
get_num_transaxial_crystals_per_block_v() const
{
    return static_cast<int>(this->crystal_repeater_y *
                            this->module_repeater_y);
}

int
InputStreamFromROOTFileForCylindricalPET::
get_num_axial_crystals_per_singles_unit() const
{
    if (this->singles_readout_depth == 1) // One PMT per Rsector
        return static_cast<int>(this->crystal_repeater_z *
                                this->module_repeater_z *
                                this->submodule_repeater_z);
    else if (this->singles_readout_depth == 2) // One PMT per module
        return static_cast<int>(this->crystal_repeater_z *
                                this->submodule_repeater_z);
    else if (this->singles_readout_depth == 3) // One PMT per submodule
        return this->crystal_repeater_z;
    else if (this->singles_readout_depth == 4) // One PMT per crystal
        return 1;
    else
        error(boost::format("Singles readout depth (%1%) is invalid") % this->singles_readout_depth);

    return 0;
}

int
InputStreamFromROOTFileForCylindricalPET::
get_num_trans_crystals_per_singles_unit() const
{
    if (this->singles_readout_depth == 1) // One PMT per Rsector
        return static_cast<int>(this->module_repeater_y *
                                this->submodule_repeater_y *
                                this->crystal_repeater_y);
    else if (this->singles_readout_depth == 2) // One PMT per module
        return static_cast<int>(this->submodule_repeater_y *
                                this->crystal_repeater_y);
    else if (this->singles_readout_depth == 3) // One PMT per submodule
        return this->crystal_repeater_y;
    else if (this->singles_readout_depth == 4) // One PMT per crystal
        return 1;
    else
        error(boost::format("Singles readout depth (%1%) is invalid") % this->singles_readout_depth);

    return 0;
}

void
InputStreamFromROOTFileForCylindricalPET::set_crystal_repeater_x(int val)
{
    crystal_repeater_x = val;
}

void
InputStreamFromROOTFileForCylindricalPET::set_crystal_repeater_y(int val)
{
    crystal_repeater_y = val;
}

void
InputStreamFromROOTFileForCylindricalPET::set_crystal_repeater_z(int val)
{
    crystal_repeater_z = val;
}

void
InputStreamFromROOTFileForCylindricalPET::set_submodule_repeater_x(int val)
{
    submodule_repeater_x = val;
}

void
InputStreamFromROOTFileForCylindricalPET::set_submodule_repeater_y(int val)
{
    submodule_repeater_y = val;
}

void
InputStreamFromROOTFileForCylindricalPET::set_submodule_repeater_z(int val)
{
    submodule_repeater_z = val;
}

void
InputStreamFromROOTFileForCylindricalPET::set_module_repeater_x(int val)
{
    module_repeater_x = val;
}

void
InputStreamFromROOTFileForCylindricalPET::set_module_repeater_y(int val)
{
    module_repeater_y = val;
}

void
InputStreamFromROOTFileForCylindricalPET::set_module_repeater_z(int val)
{
    module_repeater_z = val;
}

void
InputStreamFromROOTFileForCylindricalPET::set_rsector_repeater(int val)
{
    rsector_repeater = val;
}

END_NAMESPACE_STIR
