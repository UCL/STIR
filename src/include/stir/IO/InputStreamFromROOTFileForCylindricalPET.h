/*!
\file
\ingroup IO
\brief Declaration of class stir::InputStreamFromROOTFileForCylindricalPET

\author Nikos Efthimiou
*/
/*
    Copyright (C) 2016, UCL
    Copyright (C) 2018 University of Hull
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

#ifndef __stir_IO_InputStreamFromROOTFileForCylindricalPET_H__
#define __stir_IO_InputStreamFromROOTFileForCylindricalPET_H__

#include "stir/IO/InputStreamFromROOTFile.h"
#include "stir/RegisteredParsingObject.h"

START_NAMESPACE_STIR

/*!
  \ingroup IO
  \brief Declaration of class stir::InputStreamFromROOTFileForCylindricalPET
  \details From (<a href="http://wiki.opengatecollaboration.org/index.php/Users_Guide:Defining_a_system#CylindricalPET">here</a> ) a cylindrical PET scanner has
  five levels
    * rsector
    * module
    * submodule
    * crystal
    * layer

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
        offset (num of detectors) := 0
        low energy window (keV) := 0
        upper energy window (keV):= 10000

    End GATE_Cylindrical_PET Parameters :=
    \endverbatim

  \author Nikos Efthimiou
*/
class InputStreamFromROOTFileForCylindricalPET : public
        RegisteredParsingObject< InputStreamFromROOTFileForCylindricalPET ,
        InputStreamFromROOTFile,
        InputStreamFromROOTFile >
{
private:
    typedef RegisteredParsingObject< InputStreamFromROOTFileForCylindricalPET ,
    InputStreamFromROOTFile,
    InputStreamFromROOTFile > base_type;

public:

    //! Name which will be used when parsing a OSMAPOSLReconstruction object
    static const char * const registered_name;

    //! Default constructor
    InputStreamFromROOTFileForCylindricalPET();

    InputStreamFromROOTFileForCylindricalPET(std::string filename,
                                             std::string chain_name,
                                             int crystal_repeater_x, int crystal_repeater_y, int crystal_repeater_z,
                                             int submodule_repeater_x, int submodule_repeater_y, int submodule_repeater_z,
                                             int module_repeater_x, int module_repeater_y, int module_repeater_z,
                                             int rsector_repeater,
                                             bool exclude_scattered, bool exclude_randoms,
                                             float low_energy_window, float up_energy_window,
                                             int offset_dets);

    virtual ~InputStreamFromROOTFileForCylindricalPET() {}

    virtual
    Succeeded get_next_record(CListRecordROOT& record);
    //! Must be called before calling for the first event.
    virtual Succeeded set_up(const std::string & header_path);

    //! gives method information
    virtual std::string method_info() const;

    //! Calculate the number of rings based on the crystal, module, submodule repeaters
    inline virtual int get_num_rings() const;
    //! Calculate the number of detectors per ring based on the crystal, module, submodule repeaters
    inline virtual int get_num_dets_per_ring() const;
    //! Get the number of axial modules
    inline virtual int get_num_axial_blocks_per_bucket_v() const;
    //! Get the number of transaxial modules
    inline virtual int get_num_transaxial_blocks_per_bucket_v() const;
    //! Get the axial number of crystals per module
    inline virtual int get_num_axial_crystals_per_block_v() const;
    //! Get the transaxial number of crystals per module
    inline virtual int get_num_transaxial_crystals_per_block_v() const;
    //! Calculate the number of axial crystals per singles unit based on the repeaters numbers and the readout deptth
    inline virtual int get_num_axial_crystals_per_singles_unit() const;
    //! Calculate the number of trans crystals per singles unit based on the repeaters numbers and the readout deptth
    inline virtual int get_num_trans_crystals_per_singles_unit() const;

    inline void set_crystal_repeater_x(int&);
    inline void set_crystal_repeater_y(int&);
    inline void set_crystal_repeater_z(int&);
    inline void set_submodule_repeater_x(int&);
    inline void set_submodule_repeater_y(int&);
    inline void set_submodule_repeater_z(int&);
    inline void set_module_repeater_x(int&);
    inline void set_module_repeater_y(int&);
    inline void set_module_repeater_z(int&);
    inline void set_rsector_repeater(int&);

protected:

    virtual void set_defaults();
    virtual void initialise_keymap();
    virtual bool post_processing();

    Int_t crystalID1, crystalID2;
    Int_t submoduleID1, submoduleID2;
    Int_t moduleID1, moduleID2;
    Int_t rsectorID1, rsectorID2;

    int crystal_repeater_x;
    int crystal_repeater_y;
    int crystal_repeater_z;
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

private:
    bool check_all_required_keyword_are_set(std::string& ret);
};

END_NAMESPACE_STIR
#include "stir/IO/InputStreamFromROOTFileForCylindricalPET.inl"
#endif
