/*!
  \file
  \ingroup IO
  \brief Declaration of class stir::InputStreamFromROOTFileForECATPET

  \author Nikos Efthimiou
*/
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
#ifndef __stir_IO_InputStreamFromROOTFileForECATPET_H__
#define __stir_IO_InputStreamFromROOTFileForECATPET_H__

#include "stir/IO/InputStreamFromROOTFile.h"
#include "stir/RegisteredParsingObject.h"

START_NAMESPACE_STIR

class InputStreamFromROOTFileForECATPET : public
        RegisteredParsingObject < InputStreamFromROOTFileForECATPET,
        InputStreamFromROOTFile,
        InputStreamFromROOTFile >
{
private:
    typedef RegisteredParsingObject< InputStreamFromROOTFileForECATPET ,
    InputStreamFromROOTFile,
    InputStreamFromROOTFile > base_type;

public:

    //! Name which will be used when parsing a OSMAPOSLReconstruction object
    static const char * const registered_name;

    //! Default constructor
    InputStreamFromROOTFileForECATPET();

    InputStreamFromROOTFileForECATPET(std::string filename,
                                      std::string chain_name,
                                      int crystal_repeater_x, int crystal_repeater_y, int crystal_repeater_z,
                                      int blocks_repeater,
                                      bool exclude_scattered, bool exclude_randoms,
                                      float low_energy_window, float up_energy_window,
                                      int offset_dets);

    virtual ~InputStreamFromROOTFileForECATPET() {}

    virtual
    Succeeded get_next_record(CListRecordROOT& record);

    //! gives method information
    virtual std::string method_info() const;
    //! Must be called before calling for the first event.
    virtual Succeeded set_up(const std::string & header_path);

    //! Calculate the number of rings based on the crystals and blocks
    inline virtual int get_num_rings() const;
    //! Calculate the number of detectors per ring based on the crystals blocks
    inline virtual int get_num_dets_per_ring() const;
    //! Get the number of axial blocks
    inline virtual int get_num_axial_blocks_per_bucket_v() const;
    //! Get the number of transaxial blocks
    inline virtual int get_num_transaxial_blocks_per_bucket_v() const;
    //! Get the axial number of crystals per blocks
    inline virtual int get_num_axial_crystals_per_block_v() const;
    //! Get the transaxial number of crystals per block
    inline virtual int get_num_transaxial_crystals_per_block_v() const;
    //! Get the number of crystals per block
    inline virtual int get_num_axial_crystals_per_singles_unit() const;
    //! Get the number of crystals per block
    inline virtual int get_num_trans_crystals_per_singles_unit() const;

protected:

    virtual void set_defaults();
    virtual void initialise_keymap();
    virtual bool post_processing();

    Int_t blockID1, blockID2;
    Int_t crystalID1, crystalID2;

    int crystal_repeater_x;
    int crystal_repeater_y;
    int crystal_repeater_z;
    int block_repeater;

    //! In GATE, inside a block, the indeces start from the lower
    //! unit counting upwards. Therefore in order to align the
    //! crystals, between STIR and GATE we have to move half block more.
    int half_block;


};
END_NAMESPACE_STIR
#include "stir/IO/InputStreamFromROOTFileForECATPET.inl"
#endif
