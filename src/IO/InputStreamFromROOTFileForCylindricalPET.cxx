/*
    Copyright (C) 2016, 2021 UCL
    Copyright (C) 2018, University of Hull
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
#include "stir/IO/InputStreamFromROOTFileForCylindricalPET.h"
#include <TChain.h>

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
#if 0 // not used, so commented out (would need adapting since moving crystal_repeated_*)
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
    error("This constructor is incorrect"); //TODO set_defaults() will override the above

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
#endif

Succeeded
InputStreamFromROOTFileForCylindricalPET::
get_next_record(CListRecordROOT& record)
{
    int ring1, ring2, crystal1, crystal2;
    bool eof = false;

#ifdef STIR_OPENMP
#pragma omp critical(LISTMODEIO)
#endif
    {
    while(true)
    {
      if (current_position == nentries)
        {
          eof = true;
          break;
        }

      Long64_t brentry = stream_ptr->LoadTree(static_cast<Long64_t>(current_position));
      current_position ++ ;

      if (!this->check_brentry_randoms_scatter_energy_conditions(brentry))
        continue;


      // Get time information
      GetEntryCheck(br_time1->GetEntry(brentry));
      GetEntryCheck(br_time2->GetEntry(brentry));

      // Get positional ID information
      GetEntryCheck(br_crystalID1->GetEntry(brentry));
      GetEntryCheck(br_crystalID2->GetEntry(brentry));

      GetEntryCheck(br_submoduleID1->GetEntry(brentry));
      GetEntryCheck(br_submoduleID2->GetEntry(brentry));

      GetEntryCheck(br_moduleID1->GetEntry(brentry));
      GetEntryCheck(br_moduleID2->GetEntry(brentry));

      GetEntryCheck(br_rsectorID1->GetEntry(brentry));
      GetEntryCheck(br_rsectorID2->GetEntry(brentry));

      break;
    }

    ring1 = static_cast<int>(crystalID1/crystal_repeater_y)
            + static_cast<int>(submoduleID1/submodule_repeater_y)*get_num_axial_crystals_per_block_v()
            + static_cast<int>(moduleID1/module_repeater_y)*submodule_repeater_z*get_num_axial_crystals_per_block_v();

    ring2 = static_cast<int>(crystalID2/crystal_repeater_y)
            + static_cast<int>(submoduleID2/submodule_repeater_y)*get_num_axial_crystals_per_block_v()
            + static_cast<int>(moduleID2/module_repeater_y)*submodule_repeater_z*get_num_axial_crystals_per_block_v();

    crystal1 = rsectorID1  * module_repeater_y * submodule_repeater_y * get_num_transaxial_crystals_per_block_v()
            + (moduleID1%module_repeater_y) * submodule_repeater_y * get_num_transaxial_crystals_per_block_v()
            + (submoduleID1%submodule_repeater_y) * get_num_transaxial_crystals_per_block_v()
            + (crystalID1%crystal_repeater_y);

    crystal2 = rsectorID2 * module_repeater_y * submodule_repeater_y * get_num_transaxial_crystals_per_block_v()
            + (moduleID2%module_repeater_y) * submodule_repeater_y * get_num_transaxial_crystals_per_block_v()
            + (submoduleID2% submodule_repeater_y) * get_num_transaxial_crystals_per_block_v()
            + (crystalID2%crystal_repeater_y);

    // GATE counts crystal ID =0 the most negative. Therefore
    // ID = 0 should be negative, in Rsector 0 and the mid crystal ID be 0 .
#ifdef STIR_ROOT_ROTATION_AS_V4
    crystal1 -= half_block;
    crystal2 -= half_block;

    // Add offset
    crystal1 += offset_dets;
    crystal2 += offset_dets;
#endif
    }

    if(eof)
        return Succeeded::no;

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
    submodule_repeater_x = -1;
    submodule_repeater_y = -1;
    submodule_repeater_z = -1;
    module_repeater_x = -1;
    module_repeater_y = -1;
    module_repeater_z = -1;
    rsector_repeater = -1;
#ifdef STIR_ROOT_ROTATION_AS_V4
    half_block = module_repeater_y * submodule_repeater_y * crystal_repeater_y / 2  - 1;
    if (half_block < 0 )
      half_block = 0;
#else
    half_block = 0;
#endif
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

    stream_ptr->SetBranchAddress("crystalID1",&crystalID1, &br_crystalID1);
    stream_ptr->SetBranchAddress("crystalID2",&crystalID2, &br_crystalID2);
    stream_ptr->SetBranchAddress("submoduleID1",&submoduleID1, &br_submoduleID1);
    stream_ptr->SetBranchAddress("submoduleID2",&submoduleID2, &br_submoduleID2);
    stream_ptr->SetBranchAddress("moduleID1",&moduleID1, &br_moduleID1);
    stream_ptr->SetBranchAddress("moduleID2",&moduleID2, &br_moduleID2);
    stream_ptr->SetBranchAddress("rsectorID1",&rsectorID1, &br_rsectorID1);
    stream_ptr->SetBranchAddress("rsectorID2",&rsectorID2, &br_rsectorID2);

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
