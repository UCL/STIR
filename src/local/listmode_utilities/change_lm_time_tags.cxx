//
// $Id$
//
/*
    Copyright (C) 2005- $Date$, Hammersmith Imanet Ltd
    For internal use only.
*/
/*!
  \file 
  \ingroup listmode_utilities

  \brief Program to change time-tags of listmode data
 
  \author Kris Thielemans
  
  $Date$
  $Revision $
*/

#include "stir/shared_ptr.h"
#include "stir/ByteOrder.h"
#include "stir/listmode/CListRecord.h"
#include "stir/listmode/CListRecordECAT966.h"// TODO get rid of this
#include "stir/listmode/CListModeData.h"
#include <fstream>
#include <iostream>

static
void open_next(std::fstream& s, const std::string& filename_prefix, int& num)
{
  if (s.is_open())
    s.close();

  char txt[50];
  sprintf(txt, "_%d.lm", num);
  string filename = filename_prefix;
  filename += txt;
  s.open(filename.c_str(), std::ios::out | std::ios::binary);
  if (!s)
    stir::error("Error opening %s", filename.c_str());
  std::cerr << "\nOpened " << filename << std::endl;
  ++num;
}



USING_NAMESPACE_STIR





/************************ main ************************/


int main(int argc, char * argv[])
{
  
  if (argc!=4 && argc!=5) {
    std::cerr << "Usage: " << argv[0]
	      << " \\\n\t output_filename_prefix input_filename_prefix\\\n"
                 "\t time_offset_in_millisecs [output_file_number]\n"
	         "\noutput_file_number defaults to 1\n"
                 "Example arguments:\n"
                 "\t fixed_H666_lm1 H666_lm1 30000 2\n";
    exit(EXIT_FAILURE);
  }
  const std::string output_filename_prefix = argv[1];
  const std::string input_filename = argv[2];
  const unsigned long time_offset_in_millisecs = atol(argv[3]);
  int out_filename_counter = argc==5 ? atoi(argv[4]) : 1;
  
  shared_ptr<CListModeData>
    lm_data_ptr =
    CListModeData::read_from_file(input_filename);


  // go to the beginning of the binary data
  lm_data_ptr->reset();

  unsigned long size_written = 0;
  std::fstream out_file;
  open_next(out_file, output_filename_prefix, out_filename_counter);
  {      
    // loop over all events in the listmode file
    shared_ptr<CListRecord> record_sptr =
      lm_data_ptr->get_empty_record_sptr();
    CListRecord& record = *record_sptr;
    if (dynamic_cast<const CListRecordECAT966 *>(&record) == 0)
      error("Currently only works on 966 data. Code needs fixing.");

    const unsigned record_size = 
      sizeof(static_cast<const CListRecordECAT966&>(record).raw);
    const bool do_byte_swap =
      record_size>1 &&
      ByteOrder::big_endian != ByteOrder::get_native_order();

    while (true)
      {
        if (lm_data_ptr->get_next_record(record) == Succeeded::no) 
        {
          // no more events in file for some reason
          break; //get out of while loop
        }
        if (record.is_time())
        {
          const unsigned long new_time = 
	    record.time().get_time_in_millisecs() + time_offset_in_millisecs;
	  if (record.time().set_time_in_millisecs(new_time) == Succeeded::no)
	    {
	      warning("Did not succeed in changing time. Stopping");
	      break;
	    }
        }
	// WARNING: modifies record
	if (do_byte_swap)
	  ByteOrder::swap_order(static_cast<CListRecordECAT966&>(record).raw);
	out_file.write(reinterpret_cast<const char *>(&static_cast<const CListRecordECAT966&>(record).raw),
		       record_size);
	if (!out_file)
	  error("Error writing to file");
	size_written += record_size;

	if (size_written >= 1912602624UL)
	  open_next(out_file, output_filename_prefix, out_filename_counter);

      } // end of while loop over all events

  }


  return EXIT_SUCCESS;
}



