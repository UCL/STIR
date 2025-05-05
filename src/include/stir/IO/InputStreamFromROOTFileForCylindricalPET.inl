/*
    Copyright (C) 2016, 2020, UCL
    Copyright (C) 2018 University of Hull
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup IO
  \brief Implementation of class stir::InputStreamFromROOTFileForCylindricalPET

  \author Nikos Efthimiou
  \author Kris Thielemans
  \author Robbie Twyman
*/
#include "stir/error.h"
#include "stir/format.h"

START_NAMESPACE_STIR

int
InputStreamFromROOTFileForCylindricalPET::get_num_rings() const
{
  return static_cast<int>(this->crystal_repeater_z * this->module_repeater_z * this->submodule_repeater_z);
}

int
InputStreamFromROOTFileForCylindricalPET::get_num_dets_per_ring() const
{
  return static_cast<int>(this->rsector_repeater * this->module_repeater_y * this->submodule_repeater_y
                          * this->crystal_repeater_y);
}

int
InputStreamFromROOTFileForCylindricalPET::get_num_transaxial_blocks_per_bucket_v() const
{
  return this->submodule_repeater_y;
}

int
InputStreamFromROOTFileForCylindricalPET::get_num_axial_blocks_per_bucket_v() const
{
  return this->submodule_repeater_z;
}

int
InputStreamFromROOTFileForCylindricalPET::get_num_axial_crystals_per_singles_unit() const
{
  if (this->singles_readout_depth == 1) // One PMT per Rsector
    return static_cast<int>(this->crystal_repeater_z * this->module_repeater_z * this->submodule_repeater_z);
  else if (this->singles_readout_depth == 2) // One PMT per module
    return static_cast<int>(this->crystal_repeater_z * this->submodule_repeater_z);
  else if (this->singles_readout_depth == 3) // One PMT per submodule
    return this->crystal_repeater_z;
  else if (this->singles_readout_depth == 4) // One PMT per crystal
    return 1;
  else
    error(format("Singles readout depth ({}) is invalid", this->singles_readout_depth));

  return 0;
}

int
InputStreamFromROOTFileForCylindricalPET::get_num_trans_crystals_per_singles_unit() const
{
  if (this->singles_readout_depth == 1) // One PMT per Rsector
    return static_cast<int>(this->module_repeater_y * this->submodule_repeater_y * this->crystal_repeater_y);
  else if (this->singles_readout_depth == 2) // One PMT per module
    return static_cast<int>(this->submodule_repeater_y * this->crystal_repeater_y);
  else if (this->singles_readout_depth == 3) // One PMT per submodule
    return this->crystal_repeater_y;
  else if (this->singles_readout_depth == 4) // One PMT per crystal
    return 1;
  else
    error(format("Singles readout depth ({}) is invalid", this->singles_readout_depth));

  return 0;
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
