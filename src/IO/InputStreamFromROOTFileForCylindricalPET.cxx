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
#include "stir/IO/InputStreamFromROOTFileForCylindricalPET.h"

START_NAMESPACE_STIR

const char * const
InputStreamFromROOTFileForCylindricalPET::registered_name =
        "GATE_Cylindrical_PET";

InputStreamFromROOTFileForCylindricalPET::
InputStreamFromROOTFileForCylindricalPET():
    base_type()
{
    set_defaults();
}

InputStreamFromROOTFileForCylindricalPET::
InputStreamFromROOTFileForCylindricalPET(std::string _filename,
                                         std::string _chain_name,
                                         int crystal_repeater_x, int crystal_repeater_y, int crystal_repeater_z,
                                         int submodule_repeater_x, int submodule_repeater_y, int submodule_repeater_z,
                                         int module_repeater_x, int module_repeater_y, int module_repeater_z,
                                         int rsector_repeater,
                                         bool _exclude_scattered, bool _exclude_randoms,
                                         float _low_energy_window, float _up_energy_window,
                                         int _offset_dets):
    base_type(),
    crystal_repeater_x(crystal_repeater_x), crystal_repeater_y(crystal_repeater_y), crystal_repeater_z(crystal_repeater_z),
    submodule_repeater_x(submodule_repeater_x), submodule_repeater_y(submodule_repeater_y), submodule_repeater_z(submodule_repeater_z),
    module_repeater_x(module_repeater_x), module_repeater_y(module_repeater_y), module_repeater_z(module_repeater_z),
    rsector_repeater(rsector_repeater)
{
    set_defaults();

    filename = _filename;
    chain_name = _chain_name;
    exclude_scattered = _exclude_scattered;
    exclude_randoms = _exclude_randoms;
    low_energy_window = _low_energy_window;
    up_energy_window = _up_energy_window;
    offset_dets = _offset_dets;

    half_block = module_repeater_y * submodule_repeater_y * crystal_repeater_y / 2  - 1;
    if (half_block < 0 )
        half_block = 0;
}

Succeeded
InputStreamFromROOTFileForCylindricalPET::
get_next_record(CListRecordROOT& record)
{

    while(true)
    {
        if (current_position == nentries)
            return Succeeded::no;


        if (stream_ptr->GetEntry(static_cast<Long64_t>(current_position)) == 0 )
            return Succeeded::no;

        current_position ++ ;

        if ( (this->comptonphantom1 > 0 || this->comptonphantom2 > 0) && this->exclude_scattered )
            continue;
        if ( (this->eventID1 != this->eventID2) && this->exclude_randoms)
            continue;
        if (this->energy1 < this->low_energy_window ||
                 this->energy1 > this->up_energy_window ||
                 this->energy2 < this->low_energy_window ||
                 this->energy2 > this->up_energy_window)
            continue;

        break;
    }

    int ring1 = static_cast<int>(crystalID1/crystal_repeater_y)
            + static_cast<int>(submoduleID1/submodule_repeater_y)*crystal_repeater_z
            + static_cast<int>(moduleID1/module_repeater_y)*submodule_repeater_z*crystal_repeater_z;

    int ring2 = static_cast<int>(crystalID2/crystal_repeater_y)
            + static_cast<int>(submoduleID2/submodule_repeater_y)*crystal_repeater_z
            + static_cast<int>(moduleID2/module_repeater_y)*submodule_repeater_z*crystal_repeater_z;

    int crystal1 = rsectorID1  * module_repeater_y * submodule_repeater_y * crystal_repeater_y
            + (moduleID1%module_repeater_y) * submodule_repeater_y * crystal_repeater_y
            + (submoduleID1%submodule_repeater_y) * crystal_repeater_y
            + (crystalID1%crystal_repeater_y);

    int crystal2 = rsectorID2 * module_repeater_y * submodule_repeater_y * crystal_repeater_y
            + (moduleID2%module_repeater_y) * submodule_repeater_y * crystal_repeater_y
            + (submoduleID2% submodule_repeater_y) * crystal_repeater_y
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
InputStreamFromROOTFileForCylindricalPET::
method_info() const
{
    std::ostringstream s;
    s << this->registered_name;
    return s.str();
}

void
InputStreamFromROOTFileForCylindricalPET::set_defaults()
{
    base_type::set_defaults();
    crystal_repeater_x = -1;
    crystal_repeater_y = -1;
    crystal_repeater_z = -1;
    submodule_repeater_x = -1;
    submodule_repeater_y = -1;
    submodule_repeater_z = -1;
    module_repeater_x = -1;
    module_repeater_y = -1;
    module_repeater_z = -1;
    rsector_repeater = -1;
}

void
InputStreamFromROOTFileForCylindricalPET::initialise_keymap()
{
    base_type::initialise_keymap();
    this->parser.add_start_key("GATE_Cylindrical_PET Parameters");
    this->parser.add_stop_key("End GATE_Cylindrical_PET Parameters");
    this->parser.add_key("number of Rsectors", &this->rsector_repeater);
    this->parser.add_key("number of modules X", &this->module_repeater_x);
    this->parser.add_key("number of modules Y", &this->module_repeater_y);
    this->parser.add_key("number of modules Z", &this->module_repeater_z);

    this->parser.add_key("number of submodules X", &this->submodule_repeater_x);
    this->parser.add_key("number of submodules Y", &this->submodule_repeater_y);
    this->parser.add_key("number of submodules Z", &this->submodule_repeater_z);

    this->parser.add_key("number of crystals X", &this->crystal_repeater_x);
    this->parser.add_key("number of crystals Y", &this->crystal_repeater_y);
    this->parser.add_key("number of crystals Z", &this->crystal_repeater_z);
}

bool InputStreamFromROOTFileForCylindricalPET::
post_processing()
{
    if (base_type::post_processing())
        return true;
    return false;
}

Succeeded
InputStreamFromROOTFileForCylindricalPET::
set_up(const std::string & header_path)
{
    if (base_type::set_up(header_path) == Succeeded::no)
        return  Succeeded::no;

    std::string missing_keywords;
    if(!check_all_required_keywords_are_set(missing_keywords))
    {
        warning(missing_keywords.c_str());
        return Succeeded::no;
    }

    stream_ptr->SetBranchAddress("crystalID1",&crystalID1);
    stream_ptr->SetBranchAddress("crystalID2",&crystalID2);
    stream_ptr->SetBranchAddress("submoduleID1",&submoduleID1);
    stream_ptr->SetBranchAddress("submoduleID2",&submoduleID2);
    stream_ptr->SetBranchAddress("moduleID1",&moduleID1);
    stream_ptr->SetBranchAddress("moduleID2",&moduleID2);
    stream_ptr->SetBranchAddress("rsectorID1",&rsectorID1);
    stream_ptr->SetBranchAddress("rsectorID2",&rsectorID2);

    nentries = static_cast<unsigned long int>(stream_ptr->GetEntries());
    if (nentries == 0)
        error("InputStreamFromROOTFileForCylindricalPET: The total number of entries in the ROOT file is zero. Abort.");

    return Succeeded::yes;
}

bool InputStreamFromROOTFileForCylindricalPET::
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
        stream << "crystal_repeater_x, ";
        ok = false;
    }

    if (crystal_repeater_z == -1)
    {
        stream << "crystal_repeater_x, ";
        ok = false;
    }

    if (submodule_repeater_x == -1)
    {
        stream << "crystal_repeater_x, ";
        ok = false;
    }

    if (submodule_repeater_y == -1)
    {
        stream << "crystal_repeater_x, ";
        ok = false;
    }

    if (submodule_repeater_z == -1)
    {
        stream << "crystal_repeater_x, ";
        ok = false;
    }

    if (module_repeater_x == -1)
    {
        stream << "crystal_repeater_x, ";
        ok = false;
    }

    if (module_repeater_y == -1)
    {
        stream << "crystal_repeater_x, ";
        ok = false;
    }

    if (module_repeater_z == -1)
    {
        stream << "crystal_repeater_x, ";
        ok = false;
    }

    if (rsector_repeater == -1)
    {
        stream << "crystal_repeater_x, ";
        ok = false;
    }

    if (!ok)
        ret = stream.str();

    return ok;
}


END_NAMESPACE_STIR
