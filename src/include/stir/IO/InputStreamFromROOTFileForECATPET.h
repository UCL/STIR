/*!
\file
\ingroup IO
\brief Declaration of class stir::InputStreamFromROOTFileForECATPET

\author Nikos Efthimiou
\author Robert Twyman
*/
/*
    Copyright (C) 2016, 2021, UCL
    Copyright (C) 2018 University of Hull
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
#ifndef __stir_IO_InputStreamFromROOTFileForECATPET_H__
#define __stir_IO_InputStreamFromROOTFileForECATPET_H__

#include "stir/IO/InputStreamFromROOTFile.h"
#include "stir/RegisteredParsingObject.h"

START_NAMESPACE_STIR

/*!
  \ingroup IO
  \brief Declaration of class stir::InputStreamFromROOTFileForECATPET

  \details The ECAT system is a simplified version of CylindricalPET.
Such scanners are based on the block detector principle.
 The blocks are organized along an annular geometry to yield multi-ring detectors.
From (<a href="http://wiki.opengatecollaboration.org/index.php/Users_Guide:Defining_a_system#Ecat">here</a> ) a ECAT PET scanner
has two levels
    * block
    * crystal

    The geometry is defined through the repeaters. In the example header file found below.
    The values in the repeaters must match the values in the simulation macro file.
    \warning In case that in the simulation a level is skipped then the repeater has to
    be set to 1.

    \verbatim
    GATE scanner type := GATE_ECAT_PET
        GATE_ECAT_PET Parameters :=
        name of data file := ${INPUT_ROOT_FILE}
        name of input TChain := Coincidences

        number of blocks Y := 1
        number of blocks Z := 1
        number of crystals X := 1
        number of crystals Y := 1
        number of crystals Z := 4

        Singles readout depth := 1
        exclude scattered events := ${EXCLUDE_SCATTERED}
        exclude random events := ${EXCLUDE_RANDOM}
        low energy window (keV) := 0
        upper energy window (keV):= 10000

    End GATE_ECAT_PET Parameters :=
    \endverbatim


  \author Nikos Efthimiou
*/
class InputStreamFromROOTFileForECATPET
    : public RegisteredParsingObject<InputStreamFromROOTFileForECATPET, InputStreamFromROOTFile, InputStreamFromROOTFile>
{
private:
  typedef RegisteredParsingObject<InputStreamFromROOTFileForECATPET, InputStreamFromROOTFile, InputStreamFromROOTFile> base_type;

public:
  //! Name which will be used when parsing a OSMAPOSLReconstruction object
  static const char* const registered_name;

  //! Default constructor
  InputStreamFromROOTFileForECATPET();

#if 0 // not used, so commented out
    InputStreamFromROOTFileForECATPET(std::string filename,
                                      std::string chain_name,
                                      int crystal_repeater_x, int crystal_repeater_y, int crystal_repeater_z,
                                      int blocks_repeater_y, int blocks_repeater_z,
                                      bool exclude_scattered, bool exclude_randoms,
                                      float low_energy_window, float up_energy_window,
                                      int offset_dets);
#endif

  ~InputStreamFromROOTFileForECATPET() override
  {}

  Succeeded get_next_record(CListRecordROOT& record) override;

  //! gives method information
  virtual std::string method_info() const;
  //! Must be called before calling for the first event.
  Succeeded set_up(const std::string& header_path) override;

  //! Calculate the number of rings based on the crystals and blocks
  inline int get_num_rings() const override;
  //! Calculate the number of detectors per ring based on the crystals blocks
  inline int get_num_dets_per_ring() const override;
  //! Get the number of axial blocks
  inline int get_num_axial_blocks_per_bucket_v() const override;
  //! Get the number of transaxial blocks
  inline int get_num_transaxial_blocks_per_bucket_v() const override;
  //! Get the number of crystals per block
  inline int get_num_axial_crystals_per_singles_unit() const override;
  //! Get the number of crystals per block
  inline int get_num_trans_crystals_per_singles_unit() const override;

  inline void set_block_repeater_y(int);
  inline void set_block_repeater_z(int);

protected:
  void set_defaults() override;
  void initialise_keymap() override;
  bool post_processing() override;

  //! \name TBranches for ECAT PET
  //@{
  TBranch* br_crystalID1 = nullptr;
  TBranch* br_crystalID2 = nullptr;
  TBranch* br_blockID1 = nullptr;
  TBranch* br_blockID2 = nullptr;
  //@}

  //! \name ROOT Variables, i.e. to hold data from each entry.
  //@{
  std::int32_t blockID1, blockID2;
  std::int32_t crystalID1, crystalID2;
  //@}

  int block_repeater_y;
  int block_repeater_z;

  //! In GATE, inside a block, the indeces start from the lower
  //! unit counting upwards. Therefore in order to align the
  //! crystals, between STIR and GATE we have to move half block more.
  int half_block;

private:
  bool check_all_required_keywords_are_set(std::string& ret) const;
};
END_NAMESPACE_STIR
#include "stir/IO/InputStreamFromROOTFileForECATPET.inl"
#endif
