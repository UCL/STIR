/*
    Copyright (C) 2003- 2011, Hammersmith Imanet Ltd
    Copyright (C) 2014, University College London
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
  \brief Implementation of class stir::CListModeDataECAT
    
  \author Kris Thielemans
*/


#include "stir/listmode/CListModeDataECAT.h"
#include "stir/listmode/CListRecordECAT966.h"
#include "stir/listmode/CListRecordECAT962.h"
#include "stir/ExamInfo.h"
#include "stir/Succeeded.h"
#include "stir/is_null_ptr.h"
#include "stir/info.h"
#include <boost/format.hpp>
#ifdef HAVE_LLN_MATRIX
#include "stir/IO/stir_ecat7.h"
#else
#error Need HAVE_LLN_MATRIX
#endif
#include "boost/static_assert.hpp"
#include <iostream>
#include <fstream>
#include <typeinfo>
#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
using std::ios;
using std::fstream;
using std::ifstream;
using std::istream;
#endif

START_NAMESPACE_STIR
START_NAMESPACE_ECAT
START_NAMESPACE_ECAT7

// compile time asserts
BOOST_STATIC_ASSERT(sizeof(CListTimeDataECAT966)==4);
BOOST_STATIC_ASSERT(sizeof(CListEventDataECAT966)==4);
BOOST_STATIC_ASSERT(sizeof(CListTimeDataECAT962)==4);
BOOST_STATIC_ASSERT(sizeof(CListEventDataECAT962)==4);

template <class CListRecordT>
CListModeDataECAT<CListRecordT>::
CListModeDataECAT(const std::string& listmode_filename_prefix)
  : listmode_filename_prefix(listmode_filename_prefix)    
{
  // initialise scanner_ptr before calling open_lm_file, as it is used in that function

  this->exam_info_sptr.reset(new ExamInfo);
  ExamInfo& exam_info(*exam_info_sptr);
  exam_info.imaging_modality = ImagingModality::PT;
  // attempt to read the .sgl file
  {
    const std::string singles_filename = listmode_filename_prefix + "_1.sgl";
    ifstream singles_file(singles_filename.c_str(), ios::binary);
    char buffer[sizeof(Main_header)];
    if (!singles_file)
      {
	warning("CListModeDataECAT: Couldn't read main_header from %s. We forge ahead anyway (assuming this is ECAT 962 data).", singles_filename.c_str());
	scanner_sptr.reset(new Scanner(Scanner::E962));
	// TODO invalidate other fields in singles header
      }
    else
      {
        Main_header singles_main_header;
	singles_file.read(buffer,
			  sizeof(singles_main_header));
	unmap_main_header(buffer, &singles_main_header);
	ecat::ecat7::find_scanner(scanner_sptr, singles_main_header);

        exam_info.start_time_in_secs_since_1970 = double(singles_main_header.scan_start_time);

        switch(singles_main_header.patient_orientation)
          {
          case FeetFirstProne:
            exam_info.patient_position = PatientPosition(PatientPosition::FFP); break;
          case HeadFirstProne:
            exam_info.patient_position = PatientPosition(PatientPosition::HFP); break;
          case FeetFirstSupine:
            exam_info.patient_position = PatientPosition(PatientPosition::FFS); break;
          case HeadFirstSupine:
            exam_info.patient_position = PatientPosition(PatientPosition::HFS); break;
          case FeetFirstRight:
            exam_info.patient_position = PatientPosition(PatientPosition::FFDR); break;
          case HeadFirstRight:
            exam_info.patient_position = PatientPosition(PatientPosition::HFDR); break;
          case FeetFirstLeft:
            exam_info.patient_position = PatientPosition(PatientPosition::FFDL); break;
          case HeadFirstLeft:
            exam_info.patient_position = PatientPosition(PatientPosition::HFDL); break;
          case UnknownOrientation:
          default:
            exam_info.patient_position = PatientPosition(PatientPosition::unknown_position); break;
          }
      }
  }

  if ((scanner_sptr->get_type() == Scanner::E966 && typeid(CListRecordT) != typeid(CListRecordECAT966)) ||
      (scanner_sptr->get_type() == Scanner::E962 && typeid(CListRecordT) != typeid(CListRecordECAT962)))
    {
      error("Data in %s is from a %s scanner, but reading with wrong type of CListModeData", 
            listmode_filename_prefix.c_str(), scanner_sptr->get_name().c_str());
    }
  else if (scanner_sptr->get_type() != Scanner::E966 && scanner_sptr->get_type() != Scanner::E962)
    {
      error("CListModeDataECAT: Unsupported scanner in %s", listmode_filename_prefix.c_str());
    }

  if (open_lm_file(1) == Succeeded::no)
    error("CListModeDataECAT: error opening the first listmode file for filename %s\n",
	  listmode_filename_prefix.c_str());
}

template <class CListRecordT>
std::string
CListModeDataECAT<CListRecordT>::
get_name() const
{
  return listmode_filename_prefix + "_1.sgl";
}


template <class CListRecordT>
shared_ptr <CListRecord> 
CListModeDataECAT<CListRecordT>::
get_empty_record_sptr() const
{
  shared_ptr<CListRecord> sptr(new CListRecordT);
  return sptr;
}

template <class CListRecordT>
Succeeded
CListModeDataECAT<CListRecordT>::
open_lm_file(unsigned int new_lm_file) const
{
  // current_lm_file and new_lm_file are 1-based
  assert(new_lm_file>0);

  if (is_null_ptr(current_lm_data_ptr) || new_lm_file != current_lm_file)
    {
      // first store saved_get_positions
      if (!is_null_ptr(current_lm_data_ptr))
	{
	  assert(current_lm_file>0);
	  if (current_lm_file>=saved_get_positions_for_each_lm_data.size())
	    saved_get_positions_for_each_lm_data.resize(current_lm_file);

	  saved_get_positions_for_each_lm_data[current_lm_file-1] =
	    current_lm_data_ptr->get_saved_get_positions();
	}

      // now open new file
      std::string filename = listmode_filename_prefix;
      char rest[50];
      sprintf(rest, "_%d.lm", new_lm_file);
      filename += rest;
      info(boost::format("CListModeDataECAT: opening file %1%") % filename);
      shared_ptr<istream> stream_ptr(new fstream(filename.c_str(), ios::in | ios::binary));
      if (!(*stream_ptr))
      {
	warning("CListModeDataECAT: cannot open file %s (probably this is perfectly ok)\n ", filename.c_str());
        return Succeeded::no;
      }
      current_lm_data_ptr.reset(
	new InputStreamWithRecords<CListRecordT, bool>(stream_ptr, 
                                                       sizeof(CListTimeDataECAT966), 
                                                       sizeof(CListTimeDataECAT966), 
                                                       ByteOrder::big_endian != ByteOrder::get_native_order()));
      current_lm_file = new_lm_file;

      // now restore saved_get_positions for this file
      if (!is_null_ptr(current_lm_data_ptr) && 
	  current_lm_file<saved_get_positions_for_each_lm_data.size())
	current_lm_data_ptr->
	  set_saved_get_positions(saved_get_positions_for_each_lm_data[current_lm_file-1]);

      return Succeeded::yes;
    }
  else
    return current_lm_data_ptr->reset();
}

/*! \todo Currently switches over to the next .lm file whenever 
    get_next_record() on the current file fails. This even happens
    when it failed not because of EOF, or if the listmode file is
    shorter than 2 GB.
*/
template <class CListRecordT>
Succeeded
CListModeDataECAT<CListRecordT>::
get_next_record(CListRecord& record_of_general_type) const
{
  CListRecordT& record = static_cast<CListRecordT&>(record_of_general_type);
  if (current_lm_data_ptr->get_next_record(record) == Succeeded::yes)
    return Succeeded::yes;
  else
  {
    // warning: do not modify current_lm_file here. This is done by open_lm_file
    // open_lm_file uses current_lm_file as well
    if (open_lm_file(current_lm_file+1) == Succeeded::yes)
      return current_lm_data_ptr->get_next_record(record);
    else
      return Succeeded::no;
  }
}



template <class CListRecordT>
Succeeded
CListModeDataECAT<CListRecordT>::
reset()
{
  // current_lm_file and new_lm_file are 1-based
  assert(current_lm_file>0);
  if (current_lm_file!=1)
    {
      return open_lm_file(1);
    }
  else
    {
      return current_lm_data_ptr->reset();
    }
}


template <class CListRecordT>
CListModeData::SavedPosition
CListModeDataECAT<CListRecordT>::
save_get_position() 
{
  GetPosition current_pos;
  current_pos.first =  current_lm_file;
  current_pos.second = current_lm_data_ptr->save_get_position();
  saved_get_positions.push_back(current_pos);
  return saved_get_positions.size()-1;
} 

template <class CListRecordT>
Succeeded
CListModeDataECAT<CListRecordT>::
set_get_position(const typename CListModeDataECAT<CListRecordT>::SavedPosition& pos)
{
  assert(pos < saved_get_positions.size());
  if (open_lm_file(saved_get_positions[pos].first) == Succeeded::no)
    return Succeeded::no;

  return
    current_lm_data_ptr->set_get_position(saved_get_positions[pos].second);
}
#if 0
template <class CListRecordT>
SavedPosition
CListModeDataECAT<CListRecordT>::
save_get_pos_at_time(const double time)
{ 
  assert(time>=0);
  shared_ptr<CListRecord> record_sptr = get_empty_record_sptr(); 
  CListRecord& record = *record_sptr;

  // we first check if we can continue reading from the current point
  // or have to go back to the start of the list mode data
  {
    bool we_read_one = false;
    while (get_next_record(record) == Succeeded::yes) 
      {
	we_read_one = true;
        if(record.is_time())
	  {
	    const double new_time = record.time().get_time_in_secs();
	    if (new_time > time)
	      reset();
	    break;
	  }
      }
    if (!we_read_one)
      {
	// we might have been at the end of file
	reset();
      }
  }

  while (get_next_record(record) == Succeeded::yes) 
      {
        if(record.is_time())
	  {
	    const double new_time = record.time().get_time_in_secs();
	    if (new_time>=time)
	      {
		return save_get_position();
	      }
	  }
     }
  // TODO not nice: should flag EOF or so.
  return  save_get_position();

}

#endif
#if 0
template <class CListRecordT>
unsigned long
CListModeDataECAT<CListRecordT>::
get_num_records() const
{ 
}

#endif


// instantiations
template class CListModeDataECAT<CListRecordECAT966>;
template class CListModeDataECAT<CListRecordECAT962>;

END_NAMESPACE_ECAT7
END_NAMESPACE_ECAT
END_NAMESPACE_STIR
