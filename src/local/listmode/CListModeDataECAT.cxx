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
#include "stir/Succeeded.h"
#include "stir/is_null_ptr.h"
#include <iostream>
#include <fstream>
#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
using std::ios;
using std::fstream;
#endif

START_NAMESPACE_STIR

// TODO compile time assert
// sizeof(CListTimeDataECAT966)==4
// sizeof(CListEventDataECAT966)==4

CListModeDataECAT::
CListModeDataECAT(const string& listmode_filename_prefix)
  : listmode_filename_prefix(listmode_filename_prefix)    
{
  if (open_lm_file(1) == Succeeded::no)
    error("CListModeDataECAT: error opening the first listmode file for filename %s\n",
	  listmode_filename_prefix.c_str());
  // TODO get scanner from .sgl
  scanner_ptr = new Scanner(Scanner::E966);
}



shared_ptr <CListRecord> 
CListModeDataECAT::
get_empty_record_sptr() const
{
  // TODO differentiate using scanner_ptr
  return new CListRecordECAT966;
}

Succeeded
CListModeDataECAT::
open_lm_file(unsigned int new_lm_file) const
{
  // current_lm_file and new_lm_file are 1-based
  assert(current_lm_file>0);
  assert(new_lm_file>0);

  if (is_null_ptr(current_lm_data_ptr) || new_lm_file != current_lm_file)
    {
      // first store saved_get_positions
      if (!is_null_ptr(current_lm_data_ptr))
	{
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
unsigned long
CListModeDataECAT::
get_num_records() const
{ 
}

#endif
END_NAMESPACE_STIR
