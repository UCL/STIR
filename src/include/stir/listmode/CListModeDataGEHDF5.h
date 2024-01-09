/*
    Copyright (C) 2013-2020 University College London
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
  \author Ander Biguri (generalise from Signa to RDF9)
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

//! A class that reads the listmode data for GE scanners using the RDF9 format
/*!  \ingroup listmode
    \ingroup GE
    This file format is used by GE Signa PET/MR and can be used by GE PET/CT scanners (D690 up to DMI)
    depending on software version.
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
  unsigned long first_time_stamp;
  unsigned long lm_duration_in_millisecs;
  
  Succeeded open_lm_file(); 
};

} // namespace
}
END_NAMESPACE_STIR

#endif
