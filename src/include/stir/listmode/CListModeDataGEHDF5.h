/*
    Copyright (C) 2013-2019 University College London
    Copyright (C) 2017-2019 University of Leeds
*/
/*!
  \file
  \ingroup listmode
  \ingroup GE
  \brief Declaration of class stir::GE::RDF_HDF5::CListModeDataGEHDF5
    
  \author Kris Thielemans
  \author Ottavia Bertolli
  \author Palak Wadhwa
*/

#ifndef __stir_listmode_CListModeDataGEHDF5_H__
#define __stir_listmode_CListModeDataGEHDF5_H__

#include "stir/listmode/CListModeData.h"
#include "stir/listmode/CListRecordGEHDF5.h"
#include "stir/IO/InputStreamWithRecordsFromHDF5.h"
#include "stir/shared_ptr.h"
#include <iostream>
#include <string>


START_NAMESPACE_STIR
namespace GE {
namespace RDF_HDF5 {

//! A class that reads the listmode data for GE Signa PET/MR scanners
/*!  \ingroup listmode
    \ingroup GE
    This file format is used by GE Signa PET/MR.
*/
class CListModeDataGEHDF5 : public CListModeData
{
public:
  //! Constructor taking a filename
  CListModeDataGEHDF5(const std::string& listmode_filename);

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

  //! returns \c false, as GEHDF5 listmode data does not store delayed events (and prompts)
  /*! \todo this depends on the acquisition parameters */
  virtual bool has_delayeds() const { return false; }

private:

//  shared_ptr<GEHDF5Wrapper> input_sptr;

  typedef CListRecordGEHDF5 CListRecordT;
  std::string listmode_filename;
  shared_ptr<InputStreamWithRecordsFromHDF5<CListRecordT> > current_lm_data_ptr;
  float lm_start_time;
  float lm_duration;
  
  Succeeded open_lm_file(); 
};

} // namespace
}
END_NAMESPACE_STIR

#endif
