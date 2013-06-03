//
// $Id: CListModeDataECAT.cxx,v 1.25 2012-01-09 09:04:55 kris Exp $
//
/*
    Copyright (C) 2003-2012 Hammersmith Imanet Ltd
    Copyright (C) 2013 University College London

*/
/*!
  \file
  \ingroup listmode  
  \brief Implementation of class stir::UCL::CListModeDataECAT8_32bit

  \author Kris Thielemans
      
  $Date: 2012-01-09 09:04:55 $
  $Revision: 1.25 $
*/


#include "UCL/listmode/CListModeDataECAT8_32bit.h"
#include "UCL/listmode/CListRecordECAT8_32bit.h"
#include "stir/Succeeded.h"
#include "stir/is_null_ptr.h"
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
#endif

START_NAMESPACE_STIR
namespace UCL {


CListModeDataECAT8_32bit::
CListModeDataECAT8_32bit(const std::string& listmode_filename)
  : listmode_filename(listmode_filename)
{
  // initialise scanner_ptr before calling open_lm_file, as it is used in that function

  //interfile_parser.add_key("number of projections", &this->num_views);

  // TODO interfile_parser.parse(listmode_filename.c_str());
  interfile_parser.data_file_name = "test.lm";
  this->scanner_sptr.reset(new Scanner(Scanner::Siemens_mMR));

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

std::time_t 
CListModeDataECAT8_32bit::
get_scan_start_time_in_secs_since_1970() const
{
  // TODO
  return std::time_t(-1);
}



shared_ptr <CListRecord> 
CListModeDataECAT8_32bit::
get_empty_record_sptr() const
{
  shared_ptr<CListRecord> sptr(new CListRecordT);
  return sptr;
}


Succeeded
CListModeDataECAT8_32bit::
open_lm_file()
{
    {

      // now open new file
      std::string filename = interfile_parser.data_file_name;
      cerr << "CListModeDataECAT8_32bit: opening file " << filename << endl;
      shared_ptr<istream> stream_ptr(new fstream(filename.c_str(), ios::in | ios::binary));
      if (!(*stream_ptr))
      {
	warning("CListModeDataECAT8_32bit: cannot open file %s (probably this is perfectly ok)\n ", filename.c_str());
        return Succeeded::no;
      }
      current_lm_data_ptr.reset(
				new InputStreamWithRecords<CListRecordT, bool>(stream_ptr,  4, 4,
                                                       ByteOrder::little_endian != ByteOrder::get_native_order()));

      return Succeeded::yes;
    }
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
  return current_lm_data_ptr->save_get_position();
} 


Succeeded
CListModeDataECAT8_32bit::
set_get_position(const typename CListModeDataECAT8_32bit::SavedPosition& pos)
{
  return
    current_lm_data_ptr->set_get_position(pos);
}


} // namespace UCL 

END_NAMESPACE_STIR
