/*!
  \file
  \ingroup listmode
  \brief Utility to generate a headcurve for some listmode data
  The 'headcurve' is just a list of the total number of prompts and delayeds
  in subsequent time intervals.
  The current output is a file with 4 columns:
  \verbatim
  start_time end_time  num_prompts num_delayeds
  \endverbatim

  \author Kris Thielemans
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/
#include "stir/listmode/CListModeData.h"
#include "stir/listmode/CListRecord.h"
#include "stir/shared_ptr.h"
#include "stir/Succeeded.h"
#include "stir/utilities.h"
#include <fstream>
#include <iostream>
#include <iomanip>



USING_NAMESPACE_STIR

int main(int argc, char * argv[])
{
  
  if (argc!=2 && argc!=3) {
    std::cerr << "Usage: " << argv[0] << " listmode_file [time_interval_in_secs]\n"
	 << "time_interval_in_secs defaults to 1\n";
    exit(EXIT_FAILURE);
  }
  shared_ptr<CListModeData> lm_data_ptr =
    CListModeData::read_from_file(argv[1]);
  string hc_filename = argv[1];
  add_extension(hc_filename, ".hc");
  std::ofstream headcurve(hc_filename.c_str());
  if (!headcurve)
    {
      warning("Error opening headcurve file %s\n", hc_filename.c_str());
      exit(EXIT_FAILURE);
    }

  const double interval = argc==3 ? atof(argv[2]) : 1;


  shared_ptr <CListRecord> record_sptr = lm_data_ptr->get_empty_record_sptr();
  CListRecord& record = *record_sptr;

  double current_time = 0;
  unsigned long num_prompts = 0;
  unsigned long num_delayeds = 0;
  while (true)
    {
      if (lm_data_ptr->get_next_record(record) == Succeeded::no) 
	{
	  // no more events in file for some reason
	  break; //get out of while loop
	}
      if (record.is_time())
	{
	  const double new_time = record.time().get_time_in_secs();
	  if (new_time >= current_time+interval) 
	    {
	      headcurve << std::setw(10) << current_time
			<< std::setw(10) << current_time+interval
			<< std::setw(20) << num_prompts
			<< std::setw(20) << num_delayeds
			<< '\n';
	      num_prompts=0;
	      num_delayeds=0;
	      current_time += interval;
	    }
	}
      else if (record.is_event())
	{
	  if (record.event() .is_prompt())
	    ++num_prompts;
	  else
	    ++num_delayeds;
	}
    }
  return EXIT_SUCCESS;
}
