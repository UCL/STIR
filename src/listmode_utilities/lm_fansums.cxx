//
//
/*!
  \file 
  \ingroup listmode

  \brief Program to compute detector fansums directly from listmode data
 
  \author Kris Thielemans
  
  $Revision $
*/
/*
    Copyright (C) 2002- 2011, Hammersmith Imanet Ltd
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


#include "stir/utilities.h"
#include "stir/shared_ptr.h"
#include "stir/ParsingObject.h"
#include "stir/listmode/CListRecord.h"
#include "stir/listmode/CListEventCylindricalScannerWithDiscreteDetectors.h"
#include "stir/listmode/CListModeData.h"
#include "stir/TimeFrameDefinitions.h"
#include "stir/Scanner.h"
#include "stir/Array.h"
#include "stir/IndexRange2D.h"
#include "stir/stream.h"
#include "stir/CPUTimer.h"
#include "stir/IO/read_from_file.h"

#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include <fstream>
#include <iostream>
#include <vector>

#ifndef STIR_NO_NAMESPACES
using std::fstream;
using std::ifstream;
using std::ofstream;
using std::cerr;
using std::cout;
using std::flush;
using std::endl;
using std::min;
using std::max;
using std::vector;
#endif

START_NAMESPACE_STIR


class LmFansums : public ParsingObject
{
public:

  LmFansums(const char * const par_filename);

  int max_segment_num_to_process;
  int fan_size;
  shared_ptr<CListModeData> lm_data_ptr;
  TimeFrameDefinitions frame_defs;

  void compute();
private:

  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();
  std::string input_filename;
  std::string output_filename_prefix;
  std::string frame_definition_filename;
  bool store_prompts;
  int delayed_increment;

  bool interactive;

  void write_fan_sums(const Array<2,float>& data_fan_sums, 
               const unsigned current_frame_num) const;
};

void 
LmFansums::
set_defaults()
{
  max_segment_num_to_process = -1;
  fan_size = -1;
  store_prompts = true;
  delayed_increment = -1;
  interactive=false;
}

void 
LmFansums::
initialise_keymap()
{
  parser.add_start_key("lm_fansums Parameters");
  parser.add_key("input file",&input_filename);
  parser.add_key("frame_definition file",&frame_definition_filename);
  parser.add_key("output filename prefix",&output_filename_prefix);
  parser.add_key("tangential fan_size", &fan_size);
  parser.add_key("maximum absolute segment number to process", &max_segment_num_to_process); 
  // TODO can't do this yet
  // if (CListEvent::has_delayeds())
  {
    parser.add_key("Store 'prompts'",&store_prompts);
    parser.add_key("increment to use for 'delayeds'",&delayed_increment);
  }
  parser.add_key("List event coordinates",&interactive);
  parser.add_stop_key("END");  

}


bool
LmFansums::
post_processing()
{
  lm_data_ptr =
    read_from_file<CListModeData>(input_filename);

  const int num_rings =
      lm_data_ptr->get_scanner_ptr()->get_num_rings();
  if (max_segment_num_to_process==-1)
    max_segment_num_to_process = num_rings-1;
  else
    max_segment_num_to_process =
      min(max_segment_num_to_process, num_rings-1);

  const int max_fan_size = 
    lm_data_ptr->get_scanner_ptr()->get_max_num_non_arccorrected_bins();
  if (fan_size==-1)
    fan_size = max_fan_size;
  else
    fan_size =
      min(fan_size, max_fan_size);

  frame_defs = TimeFrameDefinitions(frame_definition_filename);
  return false;
}

LmFansums::
LmFansums(const char * const par_filename)
{
  set_defaults();
  if (par_filename!=0)
    parse(par_filename) ;
  else
    ask_parameters();

}


void
LmFansums::
compute()
{

  //*********** get Scanner details
  const int num_rings = 
    lm_data_ptr->get_scanner_ptr()->get_num_rings();
  const int num_detectors_per_ring = 
    lm_data_ptr->get_scanner_ptr()->get_num_detectors_per_ring();
  

  //*********** Finally, do the real work
  
  CPUTimer timer;
  timer.start();
    
  double time_of_last_stored_event = 0;
  long num_stored_events = 0;
  Array<2,float> data_fan_sums(IndexRange2D(num_rings, num_detectors_per_ring));
  
  // go to the beginning of the binary data
  lm_data_ptr->reset();
  
  unsigned int current_frame_num = 1;
  {      
    // loop over all events in the listmode file
    shared_ptr<CListRecord> record_sptr =
      lm_data_ptr->get_empty_record_sptr();
    CListRecord& record = *record_sptr;

    bool first_event=true;

    double current_time = 0;
    while (true)
      {
        if (lm_data_ptr->get_next_record(record) == Succeeded::no) 
        {
          // no more events in file for some reason
          write_fan_sums(data_fan_sums, current_frame_num);
          break; //get out of while loop
        }
        if (record.is_time())
        {
          const double new_time = record.time().get_time_in_secs();
          if (new_time >= frame_defs.get_end_time(current_frame_num))
          {
            while (current_frame_num <= frame_defs.get_num_frames() &&
              new_time >= frame_defs.get_end_time(current_frame_num))
            {
              write_fan_sums(data_fan_sums, current_frame_num++);
              data_fan_sums.fill(0);
            }
            if (current_frame_num > frame_defs.get_num_frames())
              break; // get out of while loop
          }
          current_time = new_time;
        }
        else if (record.is_event() && frame_defs.get_start_time(current_frame_num) <= current_time)
	  {
            // do a consistency check with dynamic_cast first
            if (first_event && dynamic_cast<const CListEventCylindricalScannerWithDiscreteDetectors *>(&record.event()) == 0)
              error("Currently only works for scanners with discrete detectors.");
            first_event=false;

            // see if we increment or decrement the value in the sinogram
            const int event_increment =
              record.event().is_prompt() 
              ? ( store_prompts ? 1 : 0 ) // it's a prompt
              :  delayed_increment;//it is a delayed-coincidence event
            
            if (event_increment==0)
              continue;
                        
            DetectionPositionPair<> det_pos;
            // because of above consistency check, we can use static_cast here (saving a bit of time)
	    static_cast<const CListEventCylindricalScannerWithDiscreteDetectors&>(record.event()).
	      get_detection_position(det_pos);
            const int ra = det_pos.pos1().axial_coord();
            const int rb = det_pos.pos2().axial_coord();
            const int a = det_pos.pos1().tangential_coord();
            const int b = det_pos.pos2().tangential_coord();
	    if (abs(ra-rb)<=max_segment_num_to_process)
	      {
		const int det_num_diff =
		  (a-b+3*num_detectors_per_ring/2)%num_detectors_per_ring;
		if (det_num_diff<=fan_size/2 || 
		    det_num_diff>=num_detectors_per_ring-fan_size/2)
		  {
		    data_fan_sums[ra][a] += event_increment;
		    data_fan_sums[rb][b] += event_increment;
		    num_stored_events += event_increment;
		  }
		else
		  {
		  }
	      }
	    else
	      {
	      }
	    
	  } // end of spatial event processing
      } // end of while loop over all events

    time_of_last_stored_event = 
      max(time_of_last_stored_event,current_time); 
  }


  timer.stop();
  
  cerr << "Last stored event was recorded after time-tick at " << time_of_last_stored_event << " secs\n";
  if (current_frame_num <= frame_defs.get_num_frames())
    cerr << "Early stop due to EOF. " << endl;
  cerr << "Total number of prompts/trues/delayed stored: " << num_stored_events << endl;

  cerr << "\nThis took " << timer.value() << "s CPU time." << endl;
}


// write fan sums to file
void 
LmFansums::
write_fan_sums(const Array<2,float>& data_fan_sums, 
               const unsigned current_frame_num) const
{
  char txt[50];
  sprintf(txt, "_f%u.dat", current_frame_num);
  std::string filename = output_filename_prefix;
  filename += txt;
  ofstream out(filename.c_str());
  out << data_fan_sums;
  cerr << "Frame " << current_frame_num << " finished" << endl;
}

END_NAMESPACE_STIR


USING_NAMESPACE_STIR





/************************ main ************************/


int main(int argc, char * argv[])
{
  
  if (argc!=1 && argc!=2) {
    cerr << "Usage: " << argv[0] << " [par_file]\n";
    exit(EXIT_FAILURE);
  }
  LmFansums lm_fansums(argc==2 ? argv[1] : 0);
  lm_fansums.compute();

  return EXIT_SUCCESS;
}



