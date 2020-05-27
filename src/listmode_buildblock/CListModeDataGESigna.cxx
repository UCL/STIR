/*
    Copyright (C) 2013-2019 University College London
    Copyright (C) 2017-2018 University of Hull
    Copyright (C) 2017-2019 University of Leeds

*/
/*!
  \file
  \ingroup listmode
  \ingroup GE
  \brief Implementation of class stir::GE::RDF_HDF5::CListModeDataGESigna

  \author Kris Thielemans
  \author Ottavia Bertolli
  \author Nikos Efthimiou
  \author Palak Wadhwa
*/


#include "stir/listmode/CListModeDataGESigna.h"
#include "stir/Succeeded.h"
#include "stir/ExamInfo.h"
#include "stir/info.h"
#include "stir/IO/GEHDF5Wrapper.h"
#include "stir/Scanner.h"
#include <boost/format.hpp>
#include <iostream>
#include <fstream>

START_NAMESPACE_STIR

namespace GE {
namespace RDF_HDF5 {

CListModeDataGESigna::
CListModeDataGESigna(const std::string& listmode_filename)
  : listmode_filename(listmode_filename)
{
  if (open_lm_file() == Succeeded::no)
    error(boost::format("CListModeDataGESigna: error opening the first listmode file for filename %s") %
      listmode_filename);
 printf( "\n Success in opening the listmode\n" );
}

std::string
CListModeDataGESigna::
get_name() const
{
  return listmode_filename;
}

std::time_t
CListModeDataGESigna::
get_scan_start_time_in_secs_since_1970() const
{
  return std::time_t(-1); // TODO
}


shared_ptr <CListRecord>
CListModeDataGESigna::
get_empty_record_sptr() const
{
  shared_ptr<CListRecord> sptr(new CListRecordT);
  return sptr;
}

Succeeded
CListModeDataGESigna::
open_lm_file()
{
  info(boost::format("CListModeDataGESigna: opening file %1%") % listmode_filename);
#if 0
  shared_ptr<std::istream> stream_ptr(new std::fstream(listmode_filename.c_str(), std::ios::in | std::ios::binary));
  if (!(*stream_ptr))
    {
      return Succeeded::no;
    }
  stream_ptr->seekg(12492704); // TODO get offset from RDF. // I got it from the listmode OtB 1/09/16 5872
  current_lm_data_ptr.reset(
                            new InputStreamWithRecords<CListRecordT, bool>(stream_ptr,
                                                                           4, 16,
                                                                           ByteOrder::little_endian != ByteOrder::get_native_order()));
#else
  if(!GEHDF5Wrapper::check_GE_signature(listmode_filename))
  {
     //! \todo N.E:Write a msg
     return Succeeded::no;
  }

//  input_sptr.reset( new GEHDF5Wrapper(listmode_filename));


#endif

  //! \todo N.E: Probably can do without the GEHDF5Wrapper here.
  GEHDF5Wrapper inputFile(listmode_filename);
 // CListModeData::scaner_sptr = inputFile.get_scanner_sptr();
  shared_ptr<Scanner> tmp = inputFile.get_scanner_sptr();
  shared_ptr<ProjDataInfo> proj_data_tmp(ProjDataInfo::ProjDataInfoCTI(tmp,2,tmp->get_num_rings()-1,
  tmp->get_num_detectors_per_ring()/2,tmp->get_max_num_non_arccorrected_bins(),
  false));
  this->set_proj_data_info_sptr(proj_data_tmp);

  //! \todo N.E: Remove hard-coded sizes;
  //! \todo Check the list record size of the signature and the maximum record size.
  current_lm_data_ptr.
  reset(
        new InputStreamWithRecordsFromHDF5<CListRecordT>(listmode_filename,
                                                               6, 16));

  return Succeeded::yes;
}

Succeeded
CListModeDataGESigna::
get_next_record(CListRecord& record_of_general_type) const
{
  CListRecordT& record = static_cast<CListRecordT&>(record_of_general_type);
  return current_lm_data_ptr->get_next_record(record);
}



Succeeded
CListModeDataGESigna::
reset()
{
  return current_lm_data_ptr->reset();
}


CListModeData::SavedPosition
CListModeDataGESigna::
save_get_position()
{
  return static_cast<SavedPosition>(current_lm_data_ptr->save_get_position());
}

Succeeded
CListModeDataGESigna::
set_get_position(const CListModeDataGESigna::SavedPosition& pos)
{
  return
    current_lm_data_ptr->set_get_position(pos);
}

} // namespace
}
END_NAMESPACE_STIR
