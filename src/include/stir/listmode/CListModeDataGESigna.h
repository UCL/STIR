/*
    Copyright (C) 2013 University College London
*/
/*!
  \file
  \ingroup listmode
  \brief Declaration of class stir::CListModeDataGESigna
    
  \author Kris Thielemans
*/

#ifndef __stir_listmode_CListModeDataGESigna_H__
#define __stir_listmode_CListModeDataGESigna_H__

#include "stir/listmode/CListModeData.h"
#include "stir/listmode/CListRecordGESigna.h"
#include "stir/IO/InputStreamWithRecordsFromHDF5.h"
#include "stir/IO/GEHDF5Data.h"
#include "stir/shared_ptr.h"
#include <iostream>
#include <string>


START_NAMESPACE_STIR

//! A class that reads the listmode data for GE Signa PET/MR scanners
/*!  \ingroup listmode
    This file format is used by GE Signa PET/MR.
*/
class CListModeDataGESigna : public CListModeData, private GEHDF5Data
{
public:
  //! Constructor taking a filename
  CListModeDataGESigna(const std::string& listmode_filename);

  virtual std::string
    get_name() const;

  virtual shared_ptr<stir::ProjDataInfo>     
    get_proj_data_info_sptr() const;

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

  //! returns \c false, as GESigna listmode data does not store delayed events (and prompts)
  /*! \todo this depends on the acquisition parameters */
  virtual bool has_delayeds() const { return false; }

private:
  typedef CListRecordGESigna CListRecordT;
  std::string listmode_filename;
  shared_ptr<stir::ProjDataInfo> proj_data_info_sptr;
  shared_ptr<InputStreamWithRecordsFromHDF5<CListRecordT> > current_lm_data_ptr;
  float lm_start_time;
  float lm_duration;
  
  Succeeded open_lm_file(); 
};

END_NAMESPACE_STIR

#endif
