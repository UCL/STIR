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

#include "stir/IO/InputStreamFromROOTFileForECATPET.h"

START_NAMESPACE_STIR

const char * const
InputStreamFromROOTFileForECATPET::registered_name =
        "GATE_ECAT_PET";

InputStreamFromROOTFileForECATPET::
InputStreamFromROOTFileForECATPET():
    base_type()
{}

InputStreamFromROOTFileForECATPET::
InputStreamFromROOTFileForECATPET(std::string _filename,
                                  std::string _chain_name,
                                  int crystal_repeater_x, int crystal_repeater_y, int crystal_repeater_z,
                                  int block_repeater,
                                  bool _exclude_scattered, bool _exclude_randoms,
                                  float _low_energy_window, float _up_energy_window,
                                  int _offset_dets):
    base_type(),
    crystal_repeater_x(crystal_repeater_x), crystal_repeater_y(crystal_repeater_y), crystal_repeater_z(crystal_repeater_z),
    block_repeater(block_repeater)
{
    filename = _filename;
    chain_name = _chain_name;
    exclude_scattered = _exclude_scattered;
    exclude_randoms = _exclude_randoms;
    low_energy_window = _low_energy_window;
    up_energy_window = _up_energy_window;
    offset_dets = _offset_dets;

    half_block = crystal_repeater_y / 2  - 1;
    if (half_block < 0 )
        half_block = 0;
}

Succeeded
InputStreamFromROOTFileForECATPET::
get_next_record(CListRecordROOT& record)
{
    while(true)
    {
        if (current_position == nentries)
            return Succeeded::no;


        if (stream_ptr->GetEntry(current_position) == 0 )
            return Succeeded::no;

        current_position ++ ;

        if ( (comptonphantom1 > 0 || comptonphantom2 > 0) && exclude_scattered )
            continue;
        if ( event1 != event2 && exclude_randoms )
            continue;
        if (energy1 < low_energy_window ||
                 energy1 > up_energy_window ||
                 energy2 < low_energy_window ||
                 energy2 > up_energy_window)
            continue;

        break;
    }

    int ring1 = static_cast<Int_t>(crystalID1/crystal_repeater_z)
            + static_cast<Int_t>(blockID1/ block_repeater)*crystal_repeater_z;

    int ring2 = static_cast<Int_t>(crystalID2/crystal_repeater_z)
            + static_cast<Int_t>(blockID2/block_repeater)*crystal_repeater_z;

    int crystal1 = (blockID1%block_repeater) * crystal_repeater_y
            + (crystalID1%crystal_repeater_z);

    int crystal2 = (blockID2%block_repeater) * crystal_repeater_y
            + (crystalID2%crystal_repeater_z);

    // GATE counts crystal ID =0 the most negative. Therefore
    // ID = 0 should be negative, in Rsector 0 and the mid crystal ID be 0 .
    crystal1 -= half_block;
    crystal2 -= half_block;

    // Add offset
    crystal1 += offset_dets;
    crystal2 += offset_dets;

    double delta_timing_bin = (time2 - time1) * least_significant_clock_bit;

    return
            record.init_from_data(ring1, ring2,
                                  crystal1, crystal2,
                                  time1, delta_timing_bin,
                                  event1, event2);
}

std::string
InputStreamFromROOTFileForECATPET::
method_info() const
{
    std::ostringstream s;
    s << this->registered_name;
    return s.str();
}

void
InputStreamFromROOTFileForECATPET::set_defaults()
{}

void
InputStreamFromROOTFileForECATPET::initialise_keymap()
{
    base_type::initialise_keymap();
    this->parser.add_start_key("GATE_ECAT_PET Parameters");
    this->parser.add_stop_key("End GATE_ECAT_PET Parameters");
    this->parser.add_key("number of blocks", &this->block_repeater);

    this->parser.add_key("number of crystals X", &this->crystal_repeater_x);
    this->parser.add_key("number of crystals Y", &this->crystal_repeater_y);
    this->parser.add_key("number of crystals Z", &this->crystal_repeater_z);
}

bool InputStreamFromROOTFileForECATPET::
post_processing()
{
    return false;
}

Succeeded InputStreamFromROOTFileForECATPET::
set_up(const std::string & header_path )
{
    if (base_type::set_up(header_path) == Succeeded::no)
        return Succeeded::no;
    stream_ptr->SetBranchAddress("crystalID1",&crystalID1);
    stream_ptr->SetBranchAddress("crystalID2",&crystalID2);
    stream_ptr->SetBranchAddress("blockID1",&blockID1);
    stream_ptr->SetBranchAddress("blockID2",&blockID2);

    nentries = static_cast<unsigned long int>(stream_ptr->GetEntries());
    if (nentries == 0)
        error("The total number of entries in the ROOT file is zero. Abort.");

    return Succeeded::yes;
}

END_NAMESPACE_STIR
