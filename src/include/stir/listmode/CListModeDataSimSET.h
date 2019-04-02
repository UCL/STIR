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

  \details This class tries to read the PHG param file, which is the
  same used to run the SimSET simulation. However, it will then check for a
  history file and initialise an InputStream.

  Unfortuantely, SimSET does not hold any information on the scanner; as name,
  model, manufacurer etc. In addition, in the case of cylindrical PET scanner
  important information as number of detectors and rings might not be present.

  THerefore in order to detect the scanner we try to harvest as much information
  as possible and compare it with the templates in Scanner.cxx.
  Scanner not included in that file will not be able to operate.

  The important bits of infomation are the number of rings, inner ring radius,
  number of detectors, number of layers and number of bins. Optionally, energy resolution.

  If two scanners share those same characteristics, the first one in order will
  be used.
  The number of rings might come from two sources. First is checked if the
  cylindrical ring geometry was separated in rings. In SimSET that is not
  mandatory.

  If that is the case, we will chack the bining parameters file.
  You can set the Z binning to be the number of rings.
  
  If the cylinder was separated in rings then, in SimSET each gap is defined
  as a ring of air. Therefore most likely the simulation will have 2N + 1 and that
  is the number with which we will be checking.

  Although the cylinder might be separated in rings, radial gaps on the cylinder
  are not defined in some way. In order to let STIR know the number of detectors per
  ring you have to set in binning parameter file the number of tagential positions
  to be half the number of detectors.

  Moreover, the the inner ring is taken by the LayerInfo->InnerRadius,

  \warning Currectly, we support only simplePET and cylindrical scanners.

  \warning When this interface was build the SimSET version was 2.9.2. But STIR
  cannot work with it. You need to download and apply the patch from
  https://gist.github.com/NikEfth/c217afccca9499f0245554941f704654.
  This patch allow SimSET to build a library to which we link to and created few
  convinience headers =).

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

    //! Check if the scanner_sptr matches the geometry in root_file_sptr.
    //! \todo This function should be extended with TOF information.
    Succeeded
    check_scanner_match_geometry(const unsigned int _numTDBins,
                                 const unsigned int _numZbins,
                                 shared_ptr<Scanner>& scanner_sptr,
                                 const double _radius = -1.0,
                                 const double _minZ = -1.0,
                                 const double _maxZ = -1.0,
                                 const unsigned int _numLayers = 1,
                                 const double _enResolution = -1.0,
                                 const double _enResReference = -1.0);

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
