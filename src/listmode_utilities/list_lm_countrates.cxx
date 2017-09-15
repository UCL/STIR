/*!
  \file
  \ingroup listmode
  \brief Utility to generate a count rate curve for some listmode data
  The count rate curve is just a list of the total number of prompts and delayeds
  in subsequent time intervals.
  The current output is a file with 4 columns:
  \verbatim
  start_time_in_secs , end_time_in_secs , num_prompts , num_delayeds
  \endverbatim

  \author Kris Thielemans
*/
/*
    Copyright (C) 2003- 2012, Hammersmith Imanet Ltd
    Copyright (C) 2017, University College London
    See STIR/LICENSE.txt for details
*/
#include "stir/listmode/CListModeData.h"
#include "stir/listmode/CListRecord.h"
#include "stir/shared_ptr.h"
#include "stir/Succeeded.h"
#include "stir/utilities.h"
#include "stir/IO/read_from_file.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>


USING_NAMESPACE_STIR

int main(int argc, char * argv[])
{
  if (argc<3 || argc>4) {
    std::cerr << "Usage: " << argv[0] << " output_filename listmode_file [time_interval_in_secs]\n"
	 << "time_interval_in_secs defaults to 1\n"
	 << "Output is a file with the count-rates per time interval in CSV format as in\n"
	 << "start_time_in_secs , end_time_in_secs , num_prompts , num_delayeds\n";
    exit(EXIT_FAILURE);
  }
  shared_ptr<CListModeData> lm_data_ptr
    (read_from_file<CListModeData>(argv[2]));
  const std::string hc_filename = argv[1];
  std::ofstream headcurve(hc_filename.c_str());
  if (!headcurve)
    {
      warning("Error opening headcurve file %s\n", hc_filename.c_str());
      exit(EXIT_FAILURE);
    }

  const double interval = argc>3 ? atof(argv[3]) : 1;

  shared_ptr <CListRecord> record_sptr = lm_data_ptr->get_empty_record_sptr();
  CListRecord& record = *record_sptr;

  double current_time = 0;
  bool first_timing_event_read=false;
  unsigned long num_prompts = 0UL;
  unsigned long num_delayeds = 0UL;
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
	  if (!first_timing_event_read)
	    {
	      current_time = new_time;
	      num_prompts=0UL;
	      num_delayeds=0UL;
	      first_timing_event_read = true;
	    }
	  else if (new_time >= current_time+interval) 
	    {
	      headcurve <<  std::fixed << std::setprecision(3)
			<< current_time
			<< " , " << current_time+interval
			<< " , " << num_prompts
			<< " , " << num_delayeds
			<< '\n';
	      num_prompts=0UL;
	      num_delayeds=0UL;
	      current_time += interval;
	    }
	}
    if (record.is_event())
	{
	  if (record.event() .is_prompt())
	    ++num_prompts;
	  else
	    ++num_delayeds;
	}
  }
  return EXIT_SUCCESS;
}
