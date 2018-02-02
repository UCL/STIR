/*
    Copyright (C) 2003-2012 Hammersmith Imanet Ltd
    Copyright (C) 2013-2014 University College London
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
  \brief Implementation of class stir::CListModeDataECAT8_32bit

  \author Kris Thielemans
*/


#include "stir/listmode/CListModeDataECAT8_32bit.h"
#include "stir/listmode/CListRecordECAT8_32bit.h"
#include "stir/ExamInfo.h"
#include "stir/Succeeded.h"
#include "stir/info.h"
#include "stir/warning.h"
#include "stir/error.h"
#include <boost/format.hpp>
#include <iostream>
#include <fstream>

START_NAMESPACE_STIR
namespace ecat {

CListModeDataECAT8_32bit::
CListModeDataECAT8_32bit(const std::string& listmode_filename)
  : listmode_filename(listmode_filename)
{
  this->interfile_parser.add_key("%axial_compression", &this->axial_compression);
  this->interfile_parser.add_key("%maximum_ring_difference", &this->maximum_ring_difference);
  this->interfile_parser.add_key("%number_of_projections", &this->number_of_projections);
  this->interfile_parser.add_key("%number_of_views", &this->number_of_views);
  this->interfile_parser.add_key("%number_of_segments", &this->number_of_segments);
  // TODO add overload to KeyParser such that we can read std::vector<int>
  // However, we would need the value of this keyword only for verification purposes, so we don't read it for now.
  // this->interfile_parser.add_key("segment_table", &this->segment_table);
  
#if 0
  // at the moment, we fix the header to confirm to "STIR interfile" as opposed to
  // "Siemens interfile", so don't enable the following.
  // It doesn't work properly anyway, as the "image duration (sec)" keyword
  // isn't "vectored" in Siemens headers (i.e. it doesn't have "[1]" appended).
  // As stir::KeyParser currently doesn't know if a keyword is vectored or not,
  // it causes memory overwrites if you use the wrong one.

  // We need to set num_time_frames to 1 as the Siemens header doesn't have the num_time_frames keyword
  {
    const int num_time_frames=1;
    this->interfile_parser.num_time_frames=1;
    this->interfile_parser.image_scaling_factors.resize(num_time_frames);
    for (int i=0; i<num_time_frames; i++)
      this->interfile_parser.image_scaling_factors[i].resize(1, 1.);
    this->interfile_parser.data_offset.resize(num_time_frames, 0UL);
    this->interfile_parser.image_relative_start_times.resize(num_time_frames, 0.);
    this->interfile_parser.image_durations.resize(num_time_frames, 0.);
  }
#endif

  this->interfile_parser.parse(listmode_filename.c_str(), false /* no warnings about unrecognised keywords */);

  this->exam_info_sptr.reset(new ExamInfo(*interfile_parser.get_exam_info_ptr()));

  const std::string originating_system(this->interfile_parser.get_exam_info_ptr()->originating_system);
  this->scanner_sptr.reset(Scanner::get_scanner_from_name(originating_system));
  if (this->scanner_sptr->get_type() == Scanner::Unknown_scanner)
    error(boost::format("Unknown value for originating_system keyword: '%s") % originating_system );

  this->proj_data_info_sptr.reset(ProjDataInfo::ProjDataInfoCTI(this->scanner_sptr, 
								this->axial_compression,
								this->maximum_ring_difference,
								this->number_of_views,
								this->number_of_projections,
								/* arc_correction*/false));

  if (this->open_lm_file() == Succeeded::no)
    error("CListModeDataECAT8_32bit: error opening the first listmode file for filename %s\n",
	  listmode_filename.c_str());
}

std::string
CListModeDataECAT8_32bit::
get_name() const
{
  return listmode_filename;
}


shared_ptr <CListRecord> 
CListModeDataECAT8_32bit::
get_empty_record_sptr() const
{
  shared_ptr<CListRecord> sptr(new CListRecordT(this->proj_data_info_sptr));
  return sptr;
}


Succeeded
CListModeDataECAT8_32bit::
open_lm_file()
{
	//const std::string filename = interfile_parser.data_file_name;
	std::string filename = interfile_parser.data_file_name;

	char directory_name[max_filename_length];
	get_directory_name(directory_name, listmode_filename.c_str());
	char full_data_file_name[max_filename_length];
	strcpy(full_data_file_name, filename.c_str());
	prepend_directory_name(full_data_file_name, directory_name);
	filename = std::string(full_data_file_name);

	info(boost::format("CListModeDataECAT8_32bit: opening file %1%") % filename);
  shared_ptr<std::istream> stream_ptr(new std::fstream(filename.c_str(), std::ios::in | std::ios::binary));
  if (!(*stream_ptr))
    {
      warning("CListModeDataECAT8_32bit: cannot open file '%s'", filename.c_str());
      return Succeeded::no;
    }
  current_lm_data_ptr.reset(
                            new InputStreamWithRecords<CListRecordT, bool>(stream_ptr,  4, 4,
                                                                           ByteOrder::little_endian != ByteOrder::get_native_order()));

  return Succeeded::yes;
}



Succeeded
CListModeDataECAT8_32bit::
get_next_record(CListRecord& record_of_general_type) const
{
  CListRecordT& record = static_cast<CListRecordT&>(record_of_general_type);
  return current_lm_data_ptr->get_next_record(record);
 }


Succeeded
CListModeDataECAT8_32bit::
reset()
{  
  return current_lm_data_ptr->reset();
}


CListModeData::SavedPosition
CListModeDataECAT8_32bit::
save_get_position() 
{
  return static_cast<SavedPosition>(current_lm_data_ptr->save_get_position());
} 


Succeeded
CListModeDataECAT8_32bit::
set_get_position(const CListModeDataECAT8_32bit::SavedPosition& pos)
{
  return
    current_lm_data_ptr->set_get_position(pos);
}

} // namespace ecat
END_NAMESPACE_STIR
