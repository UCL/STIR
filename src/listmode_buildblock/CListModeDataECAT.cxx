//
// $Id$
//
/*!
  \file
  \ingroup listmode  
  \brief Implementation of class CListModeDataECAT
    
  \author Kris Thielemans
      
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2003- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/


#include "local/stir/listmode/CListModeDataECAT.h"
#include "local/stir/listmode/CListRecordECAT966.h"
#include "local/stir/listmode/CListRecordECAT962.h"
#include "stir/Succeeded.h"
#include "stir/is_null_ptr.h"
#ifdef HAVE_LLN_MATRIX
#include "stir/IO/stir_ecat7.h"
#endif
#include <time.h>
#include <iostream>
#include <fstream>
#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
using std::ios;
using std::fstream;
using std::ifstream;
#endif

START_NAMESPACE_STIR

// TODO compile time assert
// sizeof(CListTimeDataECAT966)==4
// sizeof(CListEventDataECAT966)==4

CListModeDataECAT::
CListModeDataECAT(const string& listmode_filename_prefix)
  : listmode_filename_prefix(listmode_filename_prefix)    
{
  // initialise scanner_ptr before calling open_lm_file, as it is used in that function

#ifdef HAVE_LLN_MATRIX
  // attempt to read the .sgl file
  {
    const string singles_filename = listmode_filename_prefix + "_1.sgl";
    ifstream singles_file(singles_filename.c_str(), ios::binary);
    if (!singles_file)
      {
	warning("\nCouldn't open %s. We forge ahead anyway.\n", singles_filename.c_str());
	scanner_ptr = new Scanner(Scanner::E962);
      }
    else
      {
	char buffer[sizeof(Main_header)];
	singles_file.read(buffer,
			  sizeof(singles_main_header));
	if (!singles_file)
	  {
	    warning("\nCouldn't read main_header from %s. We forge ahead anyway.\n", singles_filename.c_str());
	    scanner_ptr = new Scanner(Scanner::E966);
	  }
	else
	  {
	    unmap_main_header(buffer, &singles_main_header);
	    ecat::ecat7::find_scanner(scanner_ptr, singles_main_header);
	    
	    time_t sec_time = singles_main_header.scan_start_time;
	    struct tm* lm_start_time_tm = localtime( &sec_time  ) ;
	    // currently use same formula as Peter
	    // it relies on TZ though: bad! (TODO)
	    lm_start_time = ( lm_start_time_tm->tm_hour * 3600.0 ) + ( lm_start_time_tm->tm_min * 60.0 ) + lm_start_time_tm->tm_sec ;
	    
	    // TODO get lm_duration from singles

	    cerr << '\n' << singles_filename << " file says that listmode start time is " << lm_start_time << endl;
	  }
      }
  }

#else
  warning("\n.sgl file not read! Assuming ECAT 966\n");
  scanner_ptr = new Scanner(Scanner::E966);
#endif

  if (open_lm_file(1) == Succeeded::no)
    error("CListModeDataECAT: error opening the first listmode file for filename %s\n",
	  listmode_filename_prefix.c_str());
}



shared_ptr <CListRecord> 
CListModeDataECAT::
get_empty_record_sptr() const
{
  // TODO differentiate using scanner_ptr
  if (scanner_ptr->get_type() == Scanner::E966)
    return new CListRecordECAT966;
  else   if (scanner_ptr->get_type() == Scanner::E962)
    return new CListRecordECAT962;
  else
    {
      error("Unsupported scanner\n");
      return 0;
    }

}

Succeeded
CListModeDataECAT::
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
      string filename = listmode_filename_prefix;
      char rest[50];
      sprintf(rest, "_%d.lm", new_lm_file);
      filename += rest;
      cerr << "CListModeDataECAT: opening file " << filename << endl;
      shared_ptr<istream> stream_ptr = 
	new fstream(filename.c_str(), ios::in | ios::binary);
      if (!(*stream_ptr))
      {
	warning("Error opening file %s\n ", filename.c_str());
        return Succeeded::no;
      }
      current_lm_data_ptr =
	new CListModeDataFromStream(stream_ptr, scanner_ptr, 
				    has_delayeds(), sizeof(CListTimeDataECAT966), get_empty_record_sptr(), 
				    ByteOrder::big_endian);
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
Succeeded
CListModeDataECAT::
get_next_record(CListRecord& record) const
{
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



Succeeded
CListModeDataECAT::
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


CListModeData::SavedPosition
CListModeDataECAT::
save_get_position() 
{
  GetPosition current_pos;
  current_pos.first =  current_lm_file;
  current_pos.second = current_lm_data_ptr->save_get_position();
  saved_get_positions.push_back(current_pos);
  return saved_get_positions.size()-1;
} 

Succeeded
CListModeDataECAT::
set_get_position(const CListModeDataECAT::SavedPosition& pos)
{
  assert(pos < saved_get_positions.size());
  if (open_lm_file(saved_get_positions[pos].first) == Succeeded::no)
    return Succeeded::no;

  return
    current_lm_data_ptr->set_get_position(saved_get_positions[pos].second);
}
#if 0
SavedPosition
CListModeDataECAT::
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
unsigned long
CListModeDataECAT::
get_num_records() const
{ 
}

#endif
END_NAMESPACE_STIR
