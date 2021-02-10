/*
    Copyright (C) 2016, UCL
    Copyright (C) 2018, University of Hull
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
#include <TChain.h>

START_NAMESPACE_STIR

const char * const
InputStreamFromROOTFileForECATPET::registered_name =
        "GATE_ECAT_PET";

InputStreamFromROOTFileForECATPET::
InputStreamFromROOTFileForECATPET():
    base_type()
{
    set_defaults();
}

#if 0 // not used, so commented out (would need adapting since moving crystal_repeated_*)
InputStreamFromROOTFileForECATPET::
InputStreamFromROOTFileForECATPET(std::string _filename,
                                  std::string _chain_name,
                                  int crystal_repeater_x, int crystal_repeater_y, int crystal_repeater_z,
                                  int block_repeater_y, int block_repeater_z,
                                  bool _exclude_scattered, bool _exclude_randoms,
                                  float _low_energy_window, float _up_energy_window,
                                  int _offset_dets):
    base_type(),
    crystal_repeater_x(crystal_repeater_x), crystal_repeater_y(crystal_repeater_y), crystal_repeater_z(crystal_repeater_z),
    block_repeater_y(block_repeater_y), block_repeater_z(block_repeater_z)
{
    set_defaults();
    error("This constructor is incorrect"); //TODO set_defaults() will override the above

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
#endif

Succeeded
InputStreamFromROOTFileForECATPET::
get_next_record(CListRecordROOT& record)
{
  while(true)
  {
    if (current_position == nentries)
      return Succeeded::no;

    auto brentry = stream_ptr->LoadTree(static_cast<Long64_t>(current_position));
    current_position ++ ;

    if (brentry < 0)
      return Succeeded::no;

    if (this->exclude_scattered) {
      // Ensure events are no scattered
      if (br_comptonPhantom1->GetEntry(brentry) == 0)
        return Succeeded::no;
      if (br_comptonPhantom2->GetEntry(brentry) == 0)
        return Succeeded::no;
      if (this->comptonphantom1 > 0 || this->comptonphantom2 > 0)
        continue;
    }

    if (this->exclude_randoms) {
      // Ensure events are from same eventID
      if (br_eventID1->GetEntry(brentry) == 0)
        return Succeeded::no;
      if (br_eventID2->GetEntry(brentry) == 0)
        return Succeeded::no;
      if ((this->eventID1 != this->eventID2))
        continue;
    }

    // Check energy information
    if (br_energy1->GetEntry(brentry) == 0)
      return Succeeded::no;
    if (br_energy2->GetEntry(brentry) == 0)
      return Succeeded::no;
    if (this->get_energy1_in_keV() < this->low_energy_window ||
        this->get_energy1_in_keV() > this->up_energy_window ||
        this->get_energy2_in_keV() < this->low_energy_window ||
        this->get_energy2_in_keV() > this->up_energy_window)
      continue;

    if (br_time1->GetEntry(brentry) == 0)
      return Succeeded::no;
    if (br_time2->GetEntry(brentry) == 0)
      return Succeeded::no;

    // Get positional ID information
    if (br_crystalID1->GetEntry(brentry) == 0)
      return Succeeded::no;
    if (br_crystalID2->GetEntry(brentry) == 0)
      return Succeeded::no;

    if (br_blockID1->GetEntry(brentry) == 0)
      return Succeeded::no;
    if (br_blockID2->GetEntry(brentry) == 0)
      return Succeeded::no;

    break;
  }

    int ring1 = static_cast<Int_t>(crystalID1/crystal_repeater_y)
            + static_cast<Int_t>(blockID1/ block_repeater_y)*crystal_repeater_z;

    int ring2 = static_cast<Int_t>(crystalID2/crystal_repeater_y)
            + static_cast<Int_t>(blockID2/block_repeater_y)*crystal_repeater_z;

    int crystal1 = (blockID1%block_repeater_y) * get_num_transaxial_crystals_per_block_v()
            + (crystalID1%crystal_repeater_y);

    int crystal2 = (blockID2%block_repeater_y) * get_num_transaxial_crystals_per_block_v()
            + (crystalID2%crystal_repeater_y);

    // GATE counts crystal ID =0 the most negative. Therefore
    // ID = 0 should be negative, in Rsector 0 and the mid crystal ID be 0 .
    crystal1 -= half_block;
    crystal2 -= half_block;

    // Add offset
    crystal1 += offset_dets;
    crystal2 += offset_dets;

    return
            record.init_from_data(ring1, ring2,
                                  crystal1, crystal2,
                                  time1, time2,
                                  eventID1, eventID2);
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
{
    base_type::set_defaults();
    block_repeater_y = -1;
    block_repeater_z = -1;
}

void
InputStreamFromROOTFileForECATPET::initialise_keymap()
{
    base_type::initialise_keymap();
    this->parser.add_start_key("GATE_ECAT_PET Parameters");
    this->parser.add_stop_key("End GATE_ECAT_PET Parameters");
    this->parser.add_key("number of blocks Y", &this->block_repeater_y);
    this->parser.add_key("number of blocks Z", &this->block_repeater_z);
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

    std::string missing_keywords;
    if(!check_all_required_keywords_are_set(missing_keywords))
    {
        warning(missing_keywords.c_str());
        return Succeeded::no;
    }

    stream_ptr->SetBranchAddress("crystalID1",&crystalID1);
    stream_ptr->SetBranchAddress("crystalID2",&crystalID2);
    stream_ptr->SetBranchAddress("blockID1",&blockID1);
    stream_ptr->SetBranchAddress("blockID2",&blockID2);

    nentries = static_cast<unsigned long int>(stream_ptr->GetEntries());
    if (nentries == 0)
        error("The total number of entries in the ROOT file is zero. Abort.");

    return Succeeded::yes;
}

bool InputStreamFromROOTFileForECATPET::
check_all_required_keywords_are_set(std::string& ret) const
{
    std::ostringstream stream("InputStreamFromROOTFileForCylindricalPET: Required keywords are missing! Check: ");
    bool ok = true;

    if (crystal_repeater_x == -1)
    {
        stream << "crystal_repeater_x, ";
        ok = false;
    }

    if (crystal_repeater_y == -1)
    {
        stream << "crystal_repeater_y, ";
        ok = false;
    }

    if (crystal_repeater_z == -1)
    {
        stream << "crystal_repeater_z, ";
        ok = false;
    }

    if (block_repeater_y == -1)
    {
        stream << "block_repeater_y, ";
        ok = false;
    }

    if (block_repeater_z == -1)
    {
        stream << "block_repeater_z, ";
        ok = false;
    }

    if (!ok)
        ret = stream.str();

    return ok;
}

END_NAMESPACE_STIR
