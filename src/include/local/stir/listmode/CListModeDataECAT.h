//
// $Id$
//
/*!
  \file
  \ingroup buildblock  
  \brief Declaration of class CListModeDataECAT
    
  \author Kris Thielemans
      
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2003- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_listmode_CListModeDataECAT_H__
#define __stir_listmode_CListModeDataECAT_H__

#include "local/stir/listmode/CListModeDataFromStream.h"
#include "stir/shared_ptr.h"

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
    of about 2 GB (to avoid problems wiht OS limits on filesize).
    In addition, there is a .sgl file with the singles per 'bucket' 
    (every second or so).
*/
class CListModeDataECAT : public CListModeData
{
public:
  //! Constructor taking a prefix for the filename
  /*! If the listmode files are called something_1.lm, something_2.lm etc.
      Then this constructor should be aclled with the argument "something" 
  */
  CListModeDataECAT(const string& listmode_filename_prefix);

  virtual 
    Succeeded get_next_record(CListRecord& record) const;

  virtual 
    Succeeded reset();

  virtual
    SavedPosition save_get_position();

  virtual
    Succeeded set_get_position(const SavedPosition&);

private:
  string listmode_filename_prefix;
  mutable unsigned int current_lm_file;
  mutable shared_ptr<CListModeDataFromStream> current_lm_data_ptr;
  vector<pair<unsigned int, SavedPosition> > saved_get_positions;
  
  // const as it modifies only mutable elements
  Succeeded open_lm_file(unsigned int) const; 
};

END_NAMESPACE_STIR

#endif
