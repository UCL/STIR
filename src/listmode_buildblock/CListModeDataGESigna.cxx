/*
    Copyright (C) 2013 University College London
*/
/*!
  \file
  \ingroup listmode  
  \brief Implementation of class stir::CListModeDataGESigna
    
  \author Kris Thielemans
*/


#include "stir/listmode/CListModeDataGESigna.h"
#include "stir/Succeeded.h"
#include "stir/ExamInfo.h"
#include "stir/info.h"
#include <boost/format.hpp>
#include <iostream>
#include <fstream>

START_NAMESPACE_STIR

CListModeDataGESigna::
CListModeDataGESigna(const std::string& listmode_filename)
  : listmode_filename(listmode_filename)    
{
  // initialise scanner_ptr before calling open_lm_file, as it is used in that function

  warning("CListModeDataGESigna: "
	  "Assuming this is GESigna STE, but couldn't find scan start time etc");
  this->scanner_sptr.reset(new Scanner(Scanner::PETMR_Signa));
  this->exam_info_sptr.reset(new ExamInfo);

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

END_NAMESPACE_STIR
