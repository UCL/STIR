//
// $Id$
//
/*!
  \file
  \ingroup listmode
  \brief Declaration of class CListModeDataECAT
    
  \author Kris Thielemans
      
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_listmode_CListModeDataECAT_H__
#define __stir_listmode_CListModeDataECAT_H__

#include "stir/listmode/CListModeDataFromStream.h"
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
using std::string;
using std::streampos;
using std::vector;
using std::pair;
#endif

START_NAMESPACE_STIR

//! A class that reads the listmode data for ECAT scanners
/*! This file format is currently used by the HR+ and HR++. It stores
    the coincidence data in multiple .lm files, with a maximum filesize
    of about 2 GB (to avoid problems with OS limits on filesize).
    In addition, there is a .sgl file with the singles per 'bucket' 
    (every second or so). The .sgl also contains a 'main_header'
    with some scanner and patient info.

    \todo This class currently relies in the fact that
     vector<>::size_type == SavedPosition
*/
class CListModeDataECAT : public CListModeData
{
public:
  //! Constructor taking a prefix for the filename
  /*! If the listmode files are called something_1.lm, something_2.lm etc.
      Then this constructor should be called with the argument "something" 

      \todo Maybe allow for passing e.g. something_2.lm in case the first lm file is missing.
  */
  CListModeDataECAT(const string& listmode_filename_prefix);

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

  //! ECAT listmode data stores delayed events as well (as opposed to prompts)
  virtual bool has_delayeds() const { return true; }

private:
  string listmode_filename_prefix;
  mutable unsigned int current_lm_file;
  mutable shared_ptr<CListModeDataFromStream> current_lm_data_ptr;
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

END_NAMESPACE_STIR

#endif
