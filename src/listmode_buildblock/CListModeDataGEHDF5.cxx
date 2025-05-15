/*
    Copyright (C) 2013-2020 University College London
    Copyright (C) 2017-2018 University of Hull
    Copyright (C) 2017-2019 University of Leeds

*/
/*!
  \file
  \ingroup listmode
  \ingroup GE
  \brief Implementation of class stir::GE::RDF_HDF5::CListModeDataGEHDF5

  \author Kris Thielemans
  \author Ottavia Bertolli
  \author Nikos Efthimiou
  \author Palak Wadhwa
*/

#include "stir/listmode/CListModeDataGEHDF5.h"
#include "stir/Succeeded.h"
#include "stir/ExamInfo.h"
#include "stir/ProjDataInfo.h"
#include "stir/info.h"
#include "stir/is_null_ptr.h"
#include "stir/IO/GEHDF5Wrapper.h"
#include "stir/Scanner.h"
#include "stir/error.h"
#include "stir/format.h"
#include <iostream>
#include <fstream>

START_NAMESPACE_STIR

namespace GE
{
namespace RDF_HDF5
{

CListModeDataGEHDF5::CListModeDataGEHDF5(const std::string& listmode_filename)
    : listmode_filename(listmode_filename)
{
  if (open_lm_file() == Succeeded::no)
    error(format("CListModeDataGEHDF5: error opening the listmode file for filename {}", listmode_filename));
}

std::string
CListModeDataGEHDF5::get_name() const
{
  return listmode_filename;
}

std::time_t
CListModeDataGEHDF5::get_scan_start_time_in_secs_since_1970() const
{
  return this->get_exam_info().start_time_in_secs_since_1970;
}

shared_ptr<CListRecord>
CListModeDataGEHDF5::get_empty_record_sptr() const
{
  if (is_null_ptr(this->get_proj_data_info_sptr()))
    error("listmode file needs to be opened before calling get_empty_record_sptr()");

  shared_ptr<CListRecord> sptr(new CListRecordT(this->get_proj_data_info_sptr(), this->first_time_stamp));
  return sptr;
}

Succeeded
CListModeDataGEHDF5::open_lm_file()
{
  info(format("CListModeDataGEHDF5: opening file {}", listmode_filename));
  if (!GEHDF5Wrapper::check_GE_signature(listmode_filename))
    {
      //! \todo N.E:Write a msg
      return Succeeded::no;
    }

  //  input_sptr.reset( new GEHDF5Wrapper(listmode_filename));

  GEHDF5Wrapper inputFile(listmode_filename);
  this->set_proj_data_info_sptr(inputFile.get_proj_data_info_sptr()->create_shared_clone());
  this->set_exam_info(*inputFile.get_exam_info_sptr());

  this->first_time_stamp = inputFile.read_dataset_uint32("/HeaderData/ListHeader/firstTmAbsTimeStamp");
  const std::uint32_t last_time_stamp = inputFile.read_dataset_uint32("/HeaderData/ListHeader/lastTmAbsTimeStamp");
  this->lm_duration_in_millisecs = last_time_stamp - this->first_time_stamp;
  info(format("First/last time-stamp: {}/{}. Duration {} ms.",
              this->first_time_stamp,
              last_time_stamp,
              this->lm_duration_in_millisecs),
       2);

  //! \todo N.E: Remove hard-coded sizes; (they're stored in GEHDF5Wrapper)

  current_lm_data_ptr.reset(new InputStreamWithRecordsFromHDF5<CListRecordT>(listmode_filename, 6, 16));

  return Succeeded::yes;
}

Succeeded
CListModeDataGEHDF5::get_next_record(CListRecord& record_of_general_type) const
{
  CListRecordT& record = static_cast<CListRecordT&>(record_of_general_type);
  return current_lm_data_ptr->get_next_record(record);
}

Succeeded
CListModeDataGEHDF5::reset()
{
  return current_lm_data_ptr->reset();
}

CListModeData::SavedPosition
CListModeDataGEHDF5::save_get_position()
{
  return static_cast<SavedPosition>(current_lm_data_ptr->save_get_position());
}

Succeeded
CListModeDataGEHDF5::set_get_position(const CListModeDataGEHDF5::SavedPosition& pos)
{
  return current_lm_data_ptr->set_get_position(pos);
}

} // namespace RDF_HDF5
} // namespace GE
END_NAMESPACE_STIR
