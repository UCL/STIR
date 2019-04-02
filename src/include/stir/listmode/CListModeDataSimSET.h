/*
    Copyright (C) 2019 University of Hull
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
/*!
  \file
  \ingroup listmode SimSET
  \brief Declaration of class stir::CListModeDataSimSET

  \details This class tries to read the PhGPhoton params files, which are the
  same as used to run the SimSET simulation. However, it will then check for a
  history file and initialise an InputStream.

  Unfortuantely, SimSET does not hold any information on the scanner as name,
  model, manufacurer etc. In addition, in the case of cylindrical PET scanner
  important information as number of detectors and rings are not present. In
  this case the only information available are radius and scanner length.
  Therefore, we will make use of the bining file setup. As the actual bining
  parameters will come from the STIR template users are encouraged to set the
  bining information such as the number of tang positions and z positions are such
  that match the Scanner num_detectors /2 and num_rings.


  \author Nikos Efthimiou
*/

#ifndef __stir_listmode_CListModeDataSimSET_H__
#define __stir_listmode_CListModeDataSimSET_H__

#include "stir/listmode/CListModeData.h"
#include "stir/listmode/CListRecordSimSET.h"
#include "stir/IO/InputStreamFromSimSET.h"
#include "stir/KeyParser.h"

START_NAMESPACE_STIR


class CListModeDataSimSET : public CListModeData
{
public:
    //! construct from the filename of the Interfile header
    CListModeDataSimSET(const std::string& _history_filename);

    //! returns the header filename
    virtual std::string
    get_name() const;
    //! Set private members default values;
    void set_defaults();

    virtual
    shared_ptr <CListRecord> get_empty_record_sptr() const;

    virtual
    Succeeded get_next_record(CListRecord& record) const;

    virtual ~CListModeDataSimSET();

    virtual
    Succeeded reset();

    virtual
    SavedPosition save_get_position();

    virtual
    Succeeded set_get_position(const SavedPosition&);

    virtual
    bool has_delayeds() const { return true; }

    virtual inline
    unsigned long int
    get_total_number_of_events() const ;

private:
    //! Check if the hroot contains a full scanner description.
    Succeeded check_scanner_definition(std::string& ret);
    //! Check if the scanner_sptr matches the geometry in root_file_sptr.
    //! \todo This function should be extended with TOF information.
    Succeeded
    check_scanner_match_geometry(const double _radius,
                                 const double _minZ,
                                 const double _maxZ,
                                 const unsigned int _numLayers,
                                 const double _enResolution,
                                 const double _enResReference,
                                 const unsigned int _numTDBins,
                                 const unsigned int _numZbins,
                                 shared_ptr<Scanner>& scanner_sptr);

    //! Pointer to the listmode data
    shared_ptr<InputStreamFromSimSET > history_file_sptr;
    //! Name of the PHG file.
    const std::string phg_filename;
    //! The name of the originating scanner
    std::string originating_system;

    Succeeded open_lm_file();
};


END_NAMESPACE_STIR

#endif
