/*!
  \file
  \ingroup IO
  \brief Declaration of class stir::InputStreamFromROOTFileForCylindricalPET

  \author Nikos Efthimiou
*/
/*
 *  Copyright (C) 2015, 2016 University of Leeds
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

#ifndef __stir_IO_InputStreamFromROOTFileForCylindricalPET_H__
#define __stir_IO_InputStreamFromROOTFileForCylindricalPET_H__

#include "stir/IO/InputStreamFromROOTFile.h"
#include "stir/RegisteredParsingObject.h"

START_NAMESPACE_STIR

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

    //! gives method information
    virtual std::string method_info() const;

    inline virtual int get_num_rings() const;

    inline virtual int get_num_dets_per_ring() const;

    //! Get the number of axial modules
    inline virtual int get_num_axial_blocks_per_bucket_v() const;
    //! Get the number of transaxial modules
    inline virtual int get_num_transaxial_blocks_per_bucket_v() const;
    //! Get the axial number of crystals per module
    inline virtual int get_num_axial_crystals_per_block_v() const;
    //! Get the transaxial number of crystals per module
    inline virtual int get_num_transaxial_crystals_per_block_v() const;

    inline virtual int get_num_axial_crystals_per_singles_unit() const;

    inline virtual int get_num_trans_crystals_per_singles_unit() const;

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
};

int
InputStreamFromROOTFileForCylindricalPET::
get_num_rings() const
{
    return static_cast<int>(this->rsector_repeater * this->module_repeater_y *
                            this->submodule_repeater_y * this->crystal_repeater_y);
}

int
InputStreamFromROOTFileForCylindricalPET::
get_num_dets_per_ring() const
{
    return static_cast<int>( this->crystal_repeater_z * this->module_repeater_z *
                             this->submodule_repeater_z);
}


int
InputStreamFromROOTFileForCylindricalPET::
get_num_axial_blocks_per_bucket_v() const
{
    return this->module_repeater_z;
}

int
InputStreamFromROOTFileForCylindricalPET::
get_num_transaxial_blocks_per_bucket_v() const
{
    return this->module_repeater_y;
}

int
InputStreamFromROOTFileForCylindricalPET::
get_num_axial_crystals_per_block_v() const
{
    return static_cast<int>(this->crystal_repeater_z *
                            this->module_repeater_z);
}

int
InputStreamFromROOTFileForCylindricalPET::
get_num_transaxial_crystals_per_block_v() const
{
    return static_cast<int>(this->crystal_repeater_y *
                            this->module_repeater_y);
}

int
InputStreamFromROOTFileForCylindricalPET::
get_num_axial_crystals_per_singles_unit() const
{
    if (this->singles_readout_depth == 1) // One PMT per Rsector
        return static_cast<int>(this->crystal_repeater_z *
                                this->module_repeater_z *
                                this->submodule_repeater_z);
    else if (this->singles_readout_depth == 2) // One PMT per module
        return static_cast<int>(this->crystal_repeater_z *
                                this->submodule_repeater_z);
    else if (this->singles_readout_depth == 3) // One PMT per submodule
        return this->crystal_repeater_z;
    else if (this->singles_readout_depth == 4) // One PMT per crystal
        return 1;
    else
        error(boost::format("Singles readout depth (%1%) is invalid") % this->singles_readout_depth);

    return 0;
}

int
InputStreamFromROOTFileForCylindricalPET::
get_num_trans_crystals_per_singles_unit() const
{
    if (this->singles_readout_depth == 1) // One PMT per Rsector
        return static_cast<int>(this->module_repeater_y *
                                this->submodule_repeater_y *
                                this->crystal_repeater_y);
    else if (this->singles_readout_depth == 2) // One PMT per module
        return static_cast<int>(this->submodule_repeater_y *
                                this->crystal_repeater_y);
    else if (this->singles_readout_depth == 3) // One PMT per submodule
        return this->crystal_repeater_y;
    else if (this->singles_readout_depth == 4) // One PMT per crystal
        return 1;
    else
        error(boost::format("Singles readout depth (%1%) is invalid") % this->singles_readout_depth);

    return 0;
}

END_NAMESPACE_STIR

#endif
