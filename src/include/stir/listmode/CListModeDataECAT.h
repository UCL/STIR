//
// $Id$
//
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd
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
  \ingroup listmode
  \brief Declaration of class stir::CListModeDataECAT
    
  \author Kris Thielemans
      
  $Date$
  $Revision$
*/

#ifndef __stir_listmode_CListModeDataECAT_H__
#define __stir_listmode_CListModeDataECAT_H__

#include "stir/listmode/CListModeData.h"
#include "stir/IO/InputStreamWithRecords.h"
#include "stir/shared_ptr.h"
#ifdef HAVE_LLN_MATRIX
#include "stir/IO/stir_ecat7.h"
#endif
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#ifndef STIR_NO_NAMESPACES
using std::istream;
using std::streampos;
using std::vector;
using std::pair;
#endif

START_NAMESPACE_STIR
START_NAMESPACE_ECAT
START_NAMESPACE_ECAT7

//! A class that reads the listmode data for ECAT scanners
/*!  \ingroup listmode
    This file format is currently used by the HR+ and HR++. It stores
    the coincidence data in multiple .lm files, with a maximum filesize
    of about 2 GB (to avoid problems with OS limits on filesize).
    In addition, there is a .sgl file with the singles rate per 'bucket'-per-ring 
    (roughly every 2 seconds). The .sgl also contains a 'main_header'
    with some scanner and patient info.

    \todo This class currently relies in the fact that
     vector<>::size_type == SavedPosition
*/
template <class CListRecordT>
class CListModeDataECAT : public CListModeData
{
public:
  //! Constructor taking a prefix for the filename
  /*! If the listmode files are called something_1.lm, something_2.lm etc.
      Then this constructor should be called with the argument "something" 

      \todo Maybe allow for passing e.g. something_2.lm in case the first lm file is missing.
  */
  CListModeDataECAT(const std::string& listmode_filename_prefix);

  virtual std::string
    get_name() const;

  virtual
    std::time_t get_scan_start_time_in_secs_since_1970() const;

  virtual 
    shared_ptr <CListRecord> get_empty_record_sptr() const;

  virtual 
    Succeeded get_next_record(CListRecord& record) const;

  virtual 
    Succeeded reset();

  virtual
    SavedPosition save_get_position();

  virtual
    Succeeded set_get_position(const SavedPosition&);

  //! returns \c true, as ECAT listmode data stores delayed events (and prompts)
  /*! \todo this might depend on the acquisition parameters */
  virtual bool has_delayeds() const { return true; }

private:
  std::string listmode_filename_prefix;
  mutable unsigned int current_lm_file;
  mutable shared_ptr<InputStreamWithRecords<CListRecordT, bool> > current_lm_data_ptr;
  //! a vector that stores the saved_get_positions for ever .lm file
  mutable vector<vector<streampos> > saved_get_positions_for_each_lm_data;
  typedef pair<unsigned int, SavedPosition> GetPosition;
  vector<GetPosition > saved_get_positions;
#ifdef HAVE_LLN_MATRIX
  Main_header singles_main_header;
#endif
  float lm_start_time;
  float lm_duration;
  
  // const as it modifies only mutable elements
  // It has to be const as e.g. get_next_record calls it
  Succeeded open_lm_file(unsigned int) const; 
};

END_NAMESPACE_ECAT7
END_NAMESPACE_ECAT
END_NAMESPACE_STIR

#endif
