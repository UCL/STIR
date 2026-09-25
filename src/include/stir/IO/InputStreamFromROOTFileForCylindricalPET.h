/*!
\file
\ingroup IO
\brief Declaration of class stir::InputStreamFromROOTFileForCylindricalPET

\author Nikos Efthimiou
\author Robert Twyman
*/
/*
 *  Copyright (C) 2016, University of Leeds
    Copyright (C) 2016, 2021, UCL
    Copyright (C) 2018 University of Hull
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#ifndef __stir_IO_InputStreamFromROOTFileForCylindricalPET_H__
#define __stir_IO_InputStreamFromROOTFileForCylindricalPET_H__

#include "stir/IO/InputStreamFromROOTFile.h"
#include "stir/RegisteredParsingObject.h"
#include "RtypesCore.h"
#include "stir/format.h"
START_NAMESPACE_STIR

/*!
  \ingroup IO
  \brief Declaration of class stir::InputStreamFromROOTFileForCylindricalPET
  \details From (<a href="http://wiki.opengatecollaboration.org/index.php/Users_Guide:Defining_a_system#CylindricalPET">here</a> )
  a cylindrical PET scanner has five levels
    * rsector
    * module
    * submodule
    * crystal
    * layer

    ## Gate 9 

    The geometry is defined through the repeaters. In the example header file found below.
    The values in the repeaters must match the values in the simulation macro file.
    \warning In case that in the simulation a level is skipped then the repeater has to
    be set to 1.

    \verbatim
    GATE scanner type := GATE_Cylindrical_PET
        GATE_Cylindrical_PET Parameters :=
        name of data file := ${INPUT_ROOT_FILE}
        name of input TChain := Coincidences

        number of Rsectors := 504
        number of modules_X := 1
        number of modules_Y := 1
        number of modules_Z := 1
        number of submodules_X := 1
        number of submodules_Y := 1
        number of submodules_Z := 1
        number of crystals_X := 1
        number of crystals_Y := 1
        number of crystals_Z := 4

        Singles readout depth := 1
        exclude scattered events := ${EXCLUDE_SCATTERED}
        exclude random events := ${EXCLUDE_RANDOM}
        low energy window (keV) := 0
        upper energy window (keV):= 10000

    End GATE_Cylindrical_PET Parameters :=
    \endverbatim

    ## Gate 10

    Gate 10 is more complicated and allows for nested repeaters that affect the orientation of the indeces. 
    Therefore we introduced the "Repeaters Desciption" a nested class that can be very flexible, and yet simplify according to 
    STIR's conventions. 
    The new class is nested to limit its scope and has arbitrary depth. 
    For example one can defince rsector/module/submodule/crystal depths.
    Bue also fullscanner/rsector/module/submodule/crystal can be used to "replicate" the structures below rsector axially, 
    for example repetition of typical PET scanner models. 

    Example shape (a 6-level hierarchy):
    \verbatim
    Repeater Description :=
      simplify upwards := <0 or 1>
      tangetial_axis := <0, 1, or 2>
      number of dimensions := <N>
      repeater type [1] := translation
      repeater level [1] := none
      !repeater size [1] := { ... }
      repeater type [2] := ring
      repeater level [2] := Rsector
      !repeater size [2] := { ... }
      repeater type [3] := translation
      repeater level [3] := module
      !repeater size [3] := { ... }
      repeater type [4] := translation
      repeater level [4] := submodule
      !repeater size [4] := { ... }
      repeater type [5] := translation
      repeater level [5] := none
      !repeater size [5] := { ... }
      repeater type [6] := translation
      repeater level [6] := crystal
      !repeater size [6] := { ... }
    End Repeater Description :=
    \endverbatim

    ### Keywords: 
    <b> number of dimentions </b> (int, req)
    How many repeater levels follow. 

    <b>repeater type [i]</b> (string: "translation" | "ring")

    <b>repeater level [i]</b> (string: "Rsector" | "module" | "submodule" | "crystal" | "none")
    "none" marks a level that exists purely for geometric accuracy in the real detector. 
     A "none"-level entry is always assumed to immediately follow (come directly after, i.e. be nested one step inside)

    <b>repeater size [i]</b> (list of 3 ints, required, "{x,y,z}")

    <b>simplify upwards</b> (bool, 0 or 1)
    For most scanners, STIR expects bucket/block structure to sit *below* Rsector. Some
  long-axial-FOV scanners instead need part of that axial structure treated as if it
  sat *above* Rsector (effectively as extra, independent axial ring segments

  <b>tangential_axis</b> (int: 0=x, 1=y)

  \author Nikos Efthimiou
*/
class InputStreamFromROOTFileForCylindricalPET
    : public RegisteredParsingObject<InputStreamFromROOTFileForCylindricalPET, InputStreamFromROOTFile, InputStreamFromROOTFile>
{
private:
  typedef RegisteredParsingObject<InputStreamFromROOTFileForCylindricalPET, InputStreamFromROOTFile, InputStreamFromROOTFile>
      base_type;

public:
  //! Name which will be used when parsing a OSMAPOSLReconstruction object
  static const char* const registered_name;

  //! Default constructor
  InputStreamFromROOTFileForCylindricalPET();

#if 0 // not used, so commented out
    InputStreamFromROOTFileForCylindricalPET(std::string filename,
                                             std::string chain_name,
                                             int crystal_repeater_x, int crystal_repeater_y, int crystal_repeater_z,
                                             int submodule_repeater_x, int submodule_repeater_y, int submodule_repeater_z,
                                             int module_repeater_x, int module_repeater_y, int module_repeater_z,
                                             int rsector_repeater,
                                             bool exclude_scattered, bool exclude_randoms,
                                             float low_energy_window, float up_energy_window,
                                             int offset_dets);
#endif

  ~InputStreamFromROOTFileForCylindricalPET() override
  {}

  Succeeded get_next_record(CListRecordROOT& record) override;
  //! Must be called before calling for the first event.
  Succeeded set_up(const std::string& header_path) override;

  //! gives method information
  virtual std::string method_info() const;

  //! Calculate the number of rings based on the crystal, module, submodule repeaters
  inline int get_num_rings() const override;
  //! Calculate the number of detectors per ring based on the crystal, module, submodule repeaters
  inline int get_num_dets_per_ring() const override;
  //! Get the number of axial modules
  inline int get_num_axial_blocks_per_bucket_v() const override;
  //! Get the number of transaxial modules
  inline int get_num_transaxial_blocks_per_bucket_v() const override;
  //! Calculate the number of axial crystals per singles unit based on the repeaters numbers and the readout deptth
  inline int get_num_axial_crystals_per_singles_unit() const override;
  //! Calculate the number of trans crystals per singles unit based on the repeaters numbers and the readout deptth
  inline int get_num_trans_crystals_per_singles_unit() const override;
  //! Get the axial number of crystals per module
  inline int get_num_axial_crystals_per_block_v() const override;
  //! Get the transaxial number of crystals per module
  inline int get_num_transaxial_crystals_per_block_v() const override;

  inline void set_submodule_repeater_x(int);
  inline void set_submodule_repeater_y(int);
  inline void set_submodule_repeater_z(int);
  inline void set_module_repeater_x(int);
  inline void set_module_repeater_y(int);
  inline void set_module_repeater_z(int);
  inline void set_rsector_repeater(int);

protected:
  void set_defaults() override;
  void initialise_keymap() override;
  bool post_processing() override;

  //! \name TBranches for Cylindrical PET
  //@{
  TBranch* br_crystalID1 = nullptr;
  TBranch* br_crystalID2 = nullptr;
  TBranch* br_submoduleID1 = nullptr;
  TBranch* br_submoduleID2 = nullptr;
  TBranch* br_moduleID1 = nullptr;
  TBranch* br_moduleID2 = nullptr;
  TBranch* br_rsectorID1 = nullptr;
  TBranch* br_rsectorID2 = nullptr;
  //! GATE 10
  TBranch* br_pre_step_uniq_vol1 = nullptr;
  TBranch* br_pre_step_uniq_vol2 = nullptr;
  //@}

  //! \name ROOT Variables, i.e. to hold data from each entry.
  //@{
  std::int32_t crystalID1, crystalID2;
  std::int32_t submoduleID1, submoduleID2;
  std::int32_t moduleID1, moduleID2;
  std::int32_t rsectorID1, rsectorID2;
  //! GATE 10
  Char_t pre_step_uniq_vol1[256], pre_step_uniq_vol2[256];
  //@}

  int submodule_repeater_x;
  int submodule_repeater_y;
  int submodule_repeater_z;
  int module_repeater_x;
  int module_repeater_y;
  int module_repeater_z;
  int rsector_repeater;

  //! In GATE, inside a block, the indeces start from the lower
  //! unit counting upwards. Therefore in order to align the
  //! crystals, between STIR and GATE we have to move half block more.
  int half_block;

  class RepeaterDescription
  {
  public:
    RepeaterDescription()
        : num_dimensions(0)
    {}

    Succeeded compute_ring_and_crystal(const std::vector<int>& ids, int& ring_out, int& crystal_out) const
    {

      if (static_cast<int>(ids.size()) < this->num_dimensions)
        {
          warning(format("compute_ring_and_crystal: 'ids' has {} entries, expected {}", ids.size(), this->num_dimensions), 3);
          return Succeeded::no;
        }

      int ring_jump = 1;
      int crystal_jump = 1;
      bool have_ring_axis = false;
      int ring_axis_value = 0;

      for (int i = this->num_dimensions - 1; i >= 0; --i)
        {
          if (this->repeater_type[i] == "ring")
            {
              if (have_ring_axis)
                error("compute_ring_and_crystal: more than one 'ring'-type repeater is not supported");
              have_ring_axis = true;
              ring_axis_value = ids[i];
              continue; // azimuthal index; folded into 'crystal' after the loop, not into 'ring'
            }

          const int sx = this->repeater_size[i].front();
          const int sz = this->repeater_size[i].back();

          ring_out += (ids[i] % sz) * ring_jump;
          ring_jump *= sz;

          crystal_out += (ids[i] / sz) * crystal_jump;
          crystal_jump *= sx;
        }

      if (have_ring_axis)
        crystal_out += ring_axis_value * crystal_jump;

      return Succeeded::yes;
    }

    std::vector<int> extract_numbers_from_string(const std::string input) const
    {
      std::vector<int> numbers;
      const auto pos = input.find("rep_");
      if (pos != std::string::npos)
        {
          std::string sub = input.substr(pos + 4);

          std::stringstream ss(sub);
          std::string token;

          while (std::getline(ss, token, '_'))
            {
              const auto dash = token.find('-');

              if (dash != std::string::npos)
                {
                  numbers.push_back(std::stoi(token.substr(0, dash)));
                  numbers.push_back(std::stoi(token.substr(dash + 1)));
                }
              else
                {
                  numbers.push_back(std::stoi(token));
                }
            }
        }
      return numbers;
    }

    Succeeded get_repeater_size_by_level(const std::string& level_name, int& out_x, int& out_y, int& out_z) const
    {
      for (int i = 0; i < this->num_dimensions; ++i)
        {
          if (this->repeater_level[i] != level_name)
            continue;
          if (this->repeater_size[i].size() != 3)
            {
              std::cerr << "RepeaterDescription: 'repeater size [" << (i + 1) << "]' (level '" << level_name
                        << "') must have exactly 3 values {x,y,z}, got " << this->repeater_size[i].size() << "\n";
              return Succeeded::no;
            }

          out_x = this->repeater_size[i][0];
          out_y = this->repeater_size[i][1];
          out_z = this->repeater_size[i][2];

          // fold in any contiguous 'none'-level repeaters that immediately follow this level
          for (int j = i + 1; j < this->num_dimensions && this->repeater_level[j] == "none"; ++j)
            {
              if (this->repeater_size[j].size() != 3)
                {
                  std::cerr << "RepeaterDescription: 'repeater size [" << (j + 1)
                            << "]' (level 'none') must have exactly 3 values {x,y,z}, got " << this->repeater_size[j].size()
                            << "\n";
                  return Succeeded::no;
                }
              out_x *= this->repeater_size[j][0];
              out_y *= this->repeater_size[j][1];
              out_z *= this->repeater_size[j][2];
            }

          return Succeeded::yes;
        }
      std::cerr << "RepeaterDescription: no entry found with 'repeater level' = '" << level_name << "'\n";
      return Succeeded::no;
    }

    Succeeded get_repeater_above_rsector(int& out_x, int& out_y, int& out_z) const
    {
      int ring_index = -1;
      for (int i = 0; i < this->num_dimensions; ++i)
        {
          if (this->repeater_type[i] == "ring")
            {
              ring_index = i;
              break;
            }
        }
      if (ring_index < 0)
        {
          std::cerr << "RepeaterDescription: no 'ring'-type (Rsector) entry found\n";
          // simplify_upwards = false;
          return Succeeded::no;
        }

      out_x = 1;
      out_y = 1;
      out_z = 1;
      for (int i = 0; i < ring_index; ++i)
        {
          if (this->repeater_size[i].size() != 3)
            {
              std::cerr << "RepeaterDescription: 'repeater size [" << (i + 1) << "]' must have exactly 3 values {x,y,z}, got "
                        << this->repeater_size[i].size() << "\n";
              // simplify_upwards = false;
              return Succeeded::no;
            }
          out_x *= this->repeater_size[i][0];
          out_y *= this->repeater_size[i][1];
          out_z *= this->repeater_size[i][2];
        }
      // simplify_upwards = true;
      return Succeeded::yes;
    }

    int get_z_repeater_above_rsector() const
    {
      int x = 1, y = 1, z = 1;
      if (get_repeater_above_rsector(x, y, z) == Succeeded::no)
        {
          std::cerr << "RepeaterDescription: get_z_repeater_above_rsector() falling back to 1\n";
          return 1;
        }
      return z;
    }

    int get_x_repeater_above_rsector() const
    {
      int x = 1, y = 1, z = 1;
      if (get_repeater_above_rsector(x, y, z) == Succeeded::no)
        {
          std::cerr << "RepeaterDescription: get_x_repeater_above_rsector() falling back to 1\n";
          return 1;
        }
      return x;
    }

    int get_y_repeater_above_rsector() const
    {
      int x = 1, y = 1, z = 1;
      if (get_repeater_above_rsector(x, y, z) == Succeeded::no)
        {
          std::cerr << "RepeaterDescription: get_y_repeater_above_rsector() falling back to 1\n";
          return 1;
        }
      return y;
    }

    static constexpr int max_dimensions = 16;
    //! tangential_axis = x: 0 , y: 1 (GATE9)
    int tangential_axis = 0;
    int num_dimensions;
    //! STIR traditionally said that module=block is the stucture immediately under the rsector.
    //! However in long scanners blocks are above rsectors. Blocks are parts of full scanners.
    //!  simplify_upwards inverst rsectors and modules(blocks)
    bool simplify_upwards;
    std::vector<std::string> repeater_type;      // e.g. "translation", "rotation", "circular"
    std::vector<std::vector<int>> repeater_size; // e.g. {1,1,8} per dimension
    std::vector<std::string> repeater_level;     // "none" | "Rsector" | "module" | "submodule" | "crystal"
  };

  RepeaterDescription repeater_description;

private:
  bool check_all_required_keywords_are_set(std::string& ret) const;
};

END_NAMESPACE_STIR
#include "stir/IO/InputStreamFromROOTFileForCylindricalPET.inl"
#endif
