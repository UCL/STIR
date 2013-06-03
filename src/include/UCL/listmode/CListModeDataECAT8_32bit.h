//
// $Id: CListModeDataECAT8_32bit.h,v 1.12 2011-06-28 14:48:09 kris Exp $
//
/*
    Copyright (C) 2003-2011 Hammersmith Imanet Ltd
    Copyright (C) 2013 University College London
*/
/*!
  \file
  \ingroup listmode
  \brief Declaration of class stir::CListModeDataECAT8_32bit
    
  \author Kris Thielemans
      
  $Date: 2011-06-28 14:48:09 $
  $Revision: 1.12 $
*/

#ifndef __stir_listmode_CListModeDataECAT8_32bit_H__
#define __stir_listmode_CListModeDataECAT8_32bit_H__

#include "stir/listmode/CListModeData.h"
#include "UCL/listmode/CListRecordECAT8_32bit.h"
#include "stir/IO/InputStreamWithRecords.h"
#include "stir/shared_ptr.h"
#include "stir/IO/InterfileHeader.h"
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
namespace UCL {

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
class CListModeDataECAT8_32bit : public CListModeData
{
public:
  //! Constructor taking a prefix for the filename
  /*! If the listmode files are called something_1.lm, something_2.lm etc.
      Then this constructor should be called with the argument "something" 

      \todo Maybe allow for passing e.g. something_2.lm in case the first lm file is missing.
  */
  CListModeDataECAT8_32bit(const std::string& listmode_filename_prefix);

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
  typedef CListRecordECAT8_32bit CListRecordT;
  std::string listmode_filename;
  shared_ptr<InputStreamWithRecords<CListRecordT, bool> > current_lm_data_ptr;
  InterfileHeader interfile_parser;
    float lm_start_time;
  float lm_duration;

  Succeeded open_lm_file();


};

} //namespace UCL 
END_NAMESPACE_STIR

#endif
