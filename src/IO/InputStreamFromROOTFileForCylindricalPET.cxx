/*
    Copyright (C) 2016, 2021 UCL
    Copyright (C) 2018, University of Hull
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
#include "stir/IO/InputStreamFromROOTFileForCylindricalPET.h"
#include <TChain.h>
#include <TLeaf.h>

#include "stir/warning.h"
#include "stir/error.h"
#include "stir/info.h"
#include "stir/format.h"

START_NAMESPACE_STIR

const char* const InputStreamFromROOTFileForCylindricalPET::registered_name = "GATE_Cylindrical_PET";

InputStreamFromROOTFileForCylindricalPET::InputStreamFromROOTFileForCylindricalPET()
    : base_type()
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

    half_block = module_repeater_y * submodule_repeater_y * crystal_repeater_y / 2 - 1;
    if (half_block < 0 )
        half_block = 0;
}
#endif

Succeeded
InputStreamFromROOTFileForCylindricalPET::get_next_record(CListRecordROOT& record)
{
  int ring1 = 0, ring2 = 0, crystal1 = 0, crystal2 = 0;
  double delta_timing_bin;
  bool eof = false;
  std::vector<int> coords1, coords2;

#ifdef STIR_OPENMP
#  pragma omp critical(LISTMODEIO)
#endif
  {
    while (true)
      {
        if (current_position == nentries)
          {
            eof = true;
            break;
          }

        Long64_t brentry = stream_ptr->LoadTree(static_cast<Long64_t>(current_position));
        current_position++;

        if (!is_gate10)
          {
            if (!this->check_brentry_randoms_scatter_energy_conditions(brentry))
              continue;

            // Get positional ID information
            GetEntryCheck(br_crystalID1->GetEntry(brentry));
            GetEntryCheck(br_crystalID2->GetEntry(brentry));

            GetEntryCheck(br_submoduleID1->GetEntry(brentry));
            GetEntryCheck(br_submoduleID2->GetEntry(brentry));

            GetEntryCheck(br_moduleID1->GetEntry(brentry));
            GetEntryCheck(br_moduleID2->GetEntry(brentry));

            GetEntryCheck(br_rsectorID1->GetEntry(brentry));
            GetEntryCheck(br_rsectorID2->GetEntry(brentry));
          }
        else
          {
            // warning("This is gate10");
            br_pre_step_uniq_vol1->GetEntry(brentry);
            br_pre_step_uniq_vol2->GetEntry(brentry);

            std::string volume1{ pre_step_uniq_vol1 };
            std::string volume2{ pre_step_uniq_vol2 };

            coords1 = repeater_description.extract_numbers_from_string(volume1);
            coords2 = repeater_description.extract_numbers_from_string(volume2);

            if (coords1.size() == 0 || coords2.size() == 0)
              {
                warning(format("InputStreamFromROOTFileForCylindricalPET: we could not extract the coordinates from {} or {}",
                               volume1,
                               volume2));
                break;
              }
          }

        // Get time information
        GetEntryCheck(br_time1->GetEntry(brentry));
        GetEntryCheck(br_time2->GetEntry(brentry));

        break;
      }

    if (!is_gate10)
      {
        ring1 = static_cast<int>(crystalID1 / crystal_repeater_y)
                + static_cast<int>(submoduleID1 / submodule_repeater_y) * get_num_axial_crystals_per_block_v()
                + static_cast<int>(moduleID1 / module_repeater_y) * submodule_repeater_z * get_num_axial_crystals_per_block_v();

        ring2 = static_cast<int>(crystalID2 / crystal_repeater_y)
                + static_cast<int>(submoduleID2 / submodule_repeater_y) * get_num_axial_crystals_per_block_v()
                + static_cast<int>(moduleID2 / module_repeater_y) * submodule_repeater_z * get_num_axial_crystals_per_block_v();

        crystal1 = rsectorID1 * module_repeater_y * submodule_repeater_y * get_num_transaxial_crystals_per_block_v()
                   + (moduleID1 % module_repeater_y) * submodule_repeater_y * get_num_transaxial_crystals_per_block_v()
                   + (submoduleID1 % submodule_repeater_y) * get_num_transaxial_crystals_per_block_v()
                   + (crystalID1 % crystal_repeater_y);

        crystal2 = rsectorID2 * module_repeater_y * submodule_repeater_y * get_num_transaxial_crystals_per_block_v()
                   + (moduleID2 % module_repeater_y) * submodule_repeater_y * get_num_transaxial_crystals_per_block_v()
                   + (submoduleID2 % submodule_repeater_y) * get_num_transaxial_crystals_per_block_v()
                   + (crystalID2 % crystal_repeater_y);

        // GATE counts crystal ID =0 the most negative. Therefore
        // ID = 0 should be negative, in Rsector 0 and the mid crystal ID be 0 .
#ifdef STIR_ROOT_ROTATION_AS_V4
        crystal1 -= half_block;
        crystal2 -= half_block;

        // Add offset
        crystal1 += offset_dets;
        crystal2 += offset_dets;
#endif

        delta_timing_bin = (time2 - time1) * least_significant_clock_bit;
      }
    else
      {
        delta_timing_bin = (time2 - time1);
        time1 /= 1e6;

        if (repeater_description.compute_ring_and_crystal(coords1, ring1, crystal1) == Succeeded::no
            || repeater_description.compute_ring_and_crystal(coords2, ring2, crystal2) == Succeeded::no)
          {
            eof = true;
          }
      }
  }

  if (eof)
    return Succeeded::no;
  return record.init_from_data(ring1, ring2, crystal1, crystal2, time1, delta_timing_bin, eventID1, eventID2);
}

std::string
InputStreamFromROOTFileForCylindricalPET::method_info() const
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
  half_block = module_repeater_y * submodule_repeater_y * crystal_repeater_y / 2 - 1;
  if (half_block < 0)
    half_block = 0;
#else
  half_block = 0;
#endif

  repeater_description.num_dimensions = 0;
  repeater_description.repeater_type.assign(RepeaterDescription::max_dimensions, std::string());
  repeater_description.repeater_size.assign(RepeaterDescription::max_dimensions, std::vector<int>());
  repeater_description.repeater_level.assign(RepeaterDescription::max_dimensions, std::string());
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

  //! This is for GATE10
  // this->parser.add_start_key("Repeater Description");
  this->parser.add_key("number of dimensions", &this->repeater_description.num_dimensions);
  this->parser.add_key("tangential axis", &this->repeater_description.tangential_axis);
  this->parser.add_vectorised_key("repeater type", &this->repeater_description.repeater_type);
  this->parser.add_vectorised_key("repeater level", &this->repeater_description.repeater_level);
  this->parser.add_vectorised_key("repeater size", &this->repeater_description.repeater_size);
  this->parser.add_key("simplify upwards", &this->repeater_description.simplify_upwards);
  // this->parser.add_stop_key("End Repeater Description");
}

bool
InputStreamFromROOTFileForCylindricalPET::post_processing()
{
  if (base_type::post_processing())
    return true;

  if (is_gate10)
    {
      auto& rd = repeater_description;

      if (rd.num_dimensions <= 0)
        {
          std::cerr << "RepeaterDescription: 'number of dimensions' must be > 0\n";
          return true;
        }
      if (rd.num_dimensions > RepeaterDescription::max_dimensions)
        {
          std::cerr << "RepeaterDescription: 'number of dimensions' (" << rd.num_dimensions << ") exceeds the maximum supported ("
                    << RepeaterDescription::max_dimensions << ")\n";
          return true;
        }

      // anything set past num_dimensions means the header disagrees with itself
      for (int i = rd.num_dimensions; i < RepeaterDescription::max_dimensions; ++i)
        {
          if (!rd.repeater_type[i].empty() || !rd.repeater_size[i].empty())
            {
              std::cerr << "RepeaterDescription: entry [" << (i + 1) << "] was set but 'number of dimensions' is only "
                        << rd.num_dimensions << "\n";
              return true;
            }
        }

      rd.repeater_type.resize(rd.num_dimensions);
      rd.repeater_size.resize(rd.num_dimensions);
      rd.repeater_level.resize(rd.num_dimensions);

      static const std::vector<std::string> allowed_levels = { "none", "Rsector", "module", "submodule", "crystal" };

      int num_named_levels = 0;
      for (int i = 0; i < rd.num_dimensions; ++i)
        {

          if (rd.repeater_size[i].empty())
            {
              std::cerr << "RepeaterDescription: 'repeater size [" << (i + 1) << "]' must have at least 1 value\n";
              return true;
            }
          if (std::find(allowed_levels.begin(), allowed_levels.end(), rd.repeater_level[i]) == allowed_levels.end())
            {
              std::cerr << "RepeaterDescription: 'repeater level [" << (i + 1) << "]' = '" << rd.repeater_level[i]
                        << "' is not one of: none, Rsector, module, submodule, crystal\n";
              return true;
            }
          if (rd.repeater_level[i] != "none")
            ++num_named_levels;
        }

      // if (num_named_levels != 4) // Rsector, module, submodule, crystal
      //   {
      //     std::cerr << "RepeaterDescription: expected exactly 4 named levels "
      //               << "(Rsector, module, submodule, crystal), found " << num_named_levels << "\n";
      //     return true;
      //   }

      for (int i = 0; i < rd.num_dimensions; ++i)
        {
          info(format("Repeater: {} with size {},{},{} assigned on level {}",
                      rd.repeater_type[i],
                      rd.repeater_size[i][0],
                      rd.repeater_size[i][1],
                      rd.repeater_size[i][2],
                      rd.repeater_level[i]));
        }

      int tmp = 0;
      rd.get_repeater_size_by_level("crystal", crystal_repeater_x, crystal_repeater_y, crystal_repeater_z);
      rd.get_repeater_size_by_level("submodule", submodule_repeater_x, submodule_repeater_y, submodule_repeater_z);
      std::cout << submodule_repeater_x << " " << submodule_repeater_y << " " << submodule_repeater_z << std::endl;
      rd.get_repeater_size_by_level("module", module_repeater_x, module_repeater_y, module_repeater_z);
      std::cout << module_repeater_x << " " << module_repeater_y << " " << module_repeater_z << std::endl;
      rd.get_repeater_size_by_level("Rsector", tmp, tmp, rsector_repeater);
      std::cout << rsector_repeater << std::endl;
    }

  return false;
}

Succeeded
InputStreamFromROOTFileForCylindricalPET::set_up(const std::string& header_path)
{
  if (base_type::set_up(header_path) == Succeeded::no)
    return Succeeded::no;

  std::string missing_keywords;
  if (!check_all_required_keywords_are_set(missing_keywords))
    {
      warning(missing_keywords.c_str());
      return Succeeded::no;
    }

  if (!is_gate10)
    {
      stream_ptr->SetBranchAddress("crystalID1", &crystalID1, &br_crystalID1);
      stream_ptr->SetBranchAddress("crystalID2", &crystalID2, &br_crystalID2);
      stream_ptr->SetBranchAddress("submoduleID1", &submoduleID1, &br_submoduleID1);
      stream_ptr->SetBranchAddress("submoduleID2", &submoduleID2, &br_submoduleID2);
      stream_ptr->SetBranchAddress("moduleID1", &moduleID1, &br_moduleID1);
      stream_ptr->SetBranchAddress("moduleID2", &moduleID2, &br_moduleID2);
      stream_ptr->SetBranchAddress("rsectorID1", &rsectorID1, &br_rsectorID1);
      stream_ptr->SetBranchAddress("rsectorID2", &rsectorID2, &br_rsectorID2);
    }
  else
    {
      stream_ptr->SetBranchAddress("PreStepUniqueVolumeID1", &pre_step_uniq_vol1, &br_pre_step_uniq_vol1);
      stream_ptr->SetBranchAddress("PreStepUniqueVolumeID2", &pre_step_uniq_vol2, &br_pre_step_uniq_vol2);
    }

  nentries = static_cast<unsigned long int>(stream_ptr->GetEntries());
  if (nentries == 0)
    error("InputStreamFromROOTFileForCylindricalPET: The total number of entries in the ROOT file is zero. Abort.");

  return Succeeded::yes;
}

bool
InputStreamFromROOTFileForCylindricalPET::check_all_required_keywords_are_set(std::string& ret) const
{
  std::ostringstream stream;
  stream << "InputStreamFromROOTFileForCylindricalPET: Required keywords are missing! Check: ";
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

  if (submodule_repeater_x == -1)
    {
      stream << "submodule_repeater_x, ";
      ok = false;
    }

  if (submodule_repeater_y == -1)
    {
      stream << "submodule_repeater_y, ";
      ok = false;
    }

  if (submodule_repeater_z == -1)
    {
      stream << "submodule_repeater_z, ";
      ok = false;
    }

  if (module_repeater_x == -1)
    {
      stream << "module_repeater_x, ";
      ok = false;
    }

  if (module_repeater_y == -1)
    {
      stream << "module_repeater_x, ";
      ok = false;
    }

  if (module_repeater_z == -1)
    {
      stream << "module_repeater_x, ";
      ok = false;
    }

  if (rsector_repeater == -1)
    {
      stream << "rsector_repeater, ";
      ok = false;
    }

  if (!ok)
    ret = stream.str();

  return ok;
}

END_NAMESPACE_STIR
