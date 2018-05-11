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
  this->interfile_parser.parse(listmode_filename.c_str());// , false /* no warnings about unrecognised keywords */);

  this->exam_info_sptr.reset(new ExamInfo(*interfile_parser.get_exam_info_ptr()));

  const std::string originating_system(this->interfile_parser.get_exam_info_ptr()->originating_system);
  shared_ptr<Scanner> this_scanner_sptr(Scanner::get_scanner_from_name(originating_system));
  if (this_scanner_sptr->get_type() == Scanner::Unknown_scanner)
    error(boost::format("Unknown value for originating_system keyword: '%s") % originating_system );

  this->set_proj_data_info_sptr(ProjDataInfo::construct_proj_data_info(this_scanner_sptr,
                                this->interfile_parser.get_axial_compression(),
                                this->interfile_parser.get_maximum_ring_difference(),
                                this->interfile_parser.get_num_views(),
                                this->interfile_parser.get_num_projections(),
                                /* arc_correction*/false)
                                 );

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
  shared_ptr<CListRecord> sptr(new CListRecordT(this->get_proj_data_info_sptr()));
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
