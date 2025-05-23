/*
 *  Copyright (C) 2015, 2016 University of Leeds
    Copyright (C) 2016, 2021 UCL
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup IO
  \brief Implementation of class stir::InputStreamFromROOTFile

  \author Nikos Efthimiou
  \author Harry Tsoumpas
  \author Robert Twyman
*/

#include "stir/IO/InputStreamFromROOTFile.h"

START_NAMESPACE_STIR

unsigned long int
InputStreamFromROOTFile::
get_total_number_of_events() const
{
    return nentries;
}

Succeeded
InputStreamFromROOTFile::
reset()
{
    current_position = starting_stream_position;
    return Succeeded::yes;
}

InputStreamFromROOTFile::SavedPosition
InputStreamFromROOTFile::
save_get_position()
{
    assert(current_position <= nentries);
    saved_get_positions.push_back(current_position);
    return saved_get_positions.size()-1;
}

Succeeded
InputStreamFromROOTFile::
set_get_position(const InputStreamFromROOTFile::SavedPosition& pos)
{
    assert(pos < saved_get_positions.size());
    if (saved_get_positions[pos] > nentries)
        current_position = nentries; // go to eof
    else
        current_position = saved_get_positions[pos];

    return Succeeded::yes;
}

std::vector<unsigned long int>
InputStreamFromROOTFile::
get_saved_get_positions() const
{
    return saved_get_positions;
}

int
InputStreamFromROOTFile::
get_number_of_energy_windows() const
{
    return num_en_windows;
}

void
InputStreamFromROOTFile::
set_saved_get_positions(const std::vector<unsigned long> &poss)
{
    saved_get_positions = poss;
}

std::vector<float>
InputStreamFromROOTFile::
get_low_energy_thres_in_keV() const
{
    std::vector<float> low_energy_window;
    low_energy_window.resize(2);
    low_energy_window[0]=1e3*low_energy_window_1;
    low_energy_window[1]=1e3*low_energy_window_2;
    return low_energy_window;
}

std::vector<float>
InputStreamFromROOTFile::
get_up_energy_thres_in_keV() const
{
    std::vector<float> up_energy_window;
    up_energy_window.resize(2);
    up_energy_window[0]=1e3*up_energy_window_1;
    up_energy_window[1]=1e3*up_energy_window_2;
    return up_energy_window;
}

std::string
InputStreamFromROOTFile::
get_ROOT_filename() const
{
    return this->filename;
}

void
InputStreamFromROOTFile::set_singles_readout_depth(int val)
{
    singles_readout_depth = val;
}

void
InputStreamFromROOTFile::set_input_filename(const std::string& val)
{
    filename = val;
}

void
InputStreamFromROOTFile::set_chain_name(const std::string& val)
{
    chain_name = val;
}

void
InputStreamFromROOTFile::set_maximum_order_of_scatter(int val)
{
    maximum_order_of_scatter = val;
}

void
InputStreamFromROOTFile::set_exclude_true_events(bool val)
{
  exclude_nonrandom = val;
}

void
InputStreamFromROOTFile::set_exclude_scattered_events(bool val)
{
  exclude_scatter = val;
}

void
InputStreamFromROOTFile::set_exclude_unscattered_events(bool val)
{
  exclude_unscattered = val;
}

void
InputStreamFromROOTFile::set_exclude_random_events(bool val)
{
    exclude_randoms = val;
}

void
InputStreamFromROOTFile::set_detectors_offset(int val)
{
    offset_dets = val;
}

void
InputStreamFromROOTFile::set_low_energy_window(const std::vector<float> &val)
{
    low_energy_window_1 = val[0];
    low_energy_window_2 = val[1];
}

void
InputStreamFromROOTFile::set_upper_energy_window(const std::vector<float> &val)
{
    up_energy_window_1 = val[0];
    up_energy_window_1 = val[1];
}

void
InputStreamFromROOTFile::set_optional_ROOT_fields(bool val)
{
    read_optional_root_fields = val;
}


int
InputStreamFromROOTFile::
get_num_axial_crystals_per_block_v() const
{
    return this->crystal_repeater_z+this->num_virtual_axial_crystals_per_block;
}

int
InputStreamFromROOTFile::
get_num_transaxial_crystals_per_block_v() const
{
    return this->crystal_repeater_y+this->num_virtual_transaxial_crystals_per_block;
}

int
InputStreamFromROOTFile::
get_num_virtual_axial_crystals_per_block() const
{
  return this->num_virtual_axial_crystals_per_block;
}
int
InputStreamFromROOTFile::
get_num_virtual_transaxial_crystals_per_block() const
{
  return this->num_virtual_transaxial_crystals_per_block;
}

END_NAMESPACE_STIR
