//
// $Id$
//
/*!
  \file 
  \ingroup utilities

  \brief Program to bin listmode data to 3d sinograms
 
  \author Kris Thielemans
  \author Sanida Mustafovic
  
  $Date$
  $Revision $
*/
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

/* Possible compilation switches:
  
HIDACREBINNER: 
  Enable code specific for the HiDAC
INCLUDE_NORMALISATION_FACTORS: 
  Enable code to include normalisation factors while rebinning.
  Currently only available for the HiDAC.
*/   

//#define HIDACREBINNER   
//#define INCLUDE_NORMALISATION_FACTORS


#include "stir/utilities.h"
#include "stir/shared_ptr.h"
#ifdef HIDACREBINNER
#include "local/stir/QHidac/lm_qhidac.h"

#else
#include "local/stir/listmode/lm.h"

#endif
#include "stir/Scanner.h"
#include "stir/Array.h"
#include "stir/IndexRange2D.h"
#include "stir/stream.h"
#include "stir/CPUTimer.h"

#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include <fstream>
#include <iostream>
#include <vector>

#ifndef STIR_NO_NAMESPACES
using std::fstream;
using std::ifstream;
using std::iostream;
using std::ofstream;
using std::streampos;
using std::cerr;
using std::cout;
using std::flush;
using std::endl;
using std::min;
using std::max;
#endif



USING_NAMESPACE_STIR



/******************** Prototypes  for local routines ************************/


/************************ main ************************/


int main(int argc, char * argv[])
{
  
  if (argc!=6) {
    cerr << "Usage: " << argv[0] << " outfilename listmode_filename  scanner_name  max_ring_diff fan_size\n";
    exit(EXIT_FAILURE);
  }

  const string input_filename = argv[2];
  const string output_filename = argv[1];

  //*********** open listmode file

  fstream input (input_filename.c_str(),ios::in | ios::binary );
  if (!input.is_open())  error("Error opening input file.\n");
  long input_file_offset = 0;
#ifdef HIDACREBINNER
  // read the list mode header to obtain lm_infos   
  LM_DATA_INFO lm_infos;
  read_lm_QHiDAC_data_head_only(&lm_infos,&input_file_offset,input_filename);
#else
  // something similar will be done for other listmode types. TODO
#endif


  //*********** get Scanner details
  shared_ptr<Scanner> scanner_ptr = Scanner::get_scanner_from_name(argv[3]);
  const int num_rings = scanner_ptr->get_num_rings();
  const int num_detectors_per_ring = scanner_ptr->get_num_detectors_per_ring();
  const int fan_size = atoi(argv[5]);
  const int max_ring_diff = atoi(argv[4]);


  //*********** ask some more questions on how to process the data


#ifdef HIDACREBINNER     
  const unsigned int max_converter = 
    ask_num("Maximum allowed converter",0,15,15);
#endif
  
#ifdef INCLUDE_NORMALISATION_FACTORS
  const bool do_normalisation = ask("Include normalisation factors?", false);
#  ifdef HIDACREBINNER     
  const bool handle_anode_wire_efficiency  =
    do_normalisation ? ask("normalise for anode wire efficiency?",false) : false;
#  endif
#endif

  const bool store_prompts = 
    CLIST_EVENT::has_delayeds() ? ask("Store 'prompts' ?",true) : true;
  const int delayed_increment = 
    CLIST_EVENT::has_delayeds() 
    ? 
    ( store_prompts ?
      (ask("Subtract 'delayed' coincidences ?",true) ? -1 : 0)
      :
      1
    )
    :
    0 /* listmode file does not store delayeds*/; 
    
  // Determine maximum number of events from file size 
  long max_num_events;
  { 
    // start of the binary part
    input.seekg(input_file_offset,ios::beg);
    const streampos start_of_stream = input.tellg();
    
    input.seekg(0, ios::end);
    const streampos end_stream_position = input.tellg();
    
    max_num_events = (long)(end_stream_position - start_of_stream) / sizeof(CLIST_EVENT);
  }


  const bool do_time_frame = true;//ask("Do time frame",true); TODO?
  long num_events_to_store;
  double start_time = 0;
  double end_time = 0;

  if (do_time_frame)
  {
    start_time = ask_num("Start time (in secs)",0.,1.e6,0.);
    end_time = ask_num("End time (in secs)",start_time,1.e6,3600.); // TODO get sensible maximum from data
    num_events_to_store = max_num_events;
  }
  else
  {
    num_events_to_store = 
      ask_num("Number of (prompt/true/random) events to store", 
      (long)0, max_num_events, max_num_events);
  }

  const bool interactive=ask("List event coordinates?",false);

  const shared_ptr<ProjDataInfoCylindricalNoArcCorr> proj_data_info_ptr =
    dynamic_cast<ProjDataInfoCylindricalNoArcCorr *>
    (ProjDataInfo::ProjDataInfoCTI(scanner_ptr, 1, max_ring_diff,
                                   scanner_ptr->get_num_detectors_per_ring()/2,
                                   fan_size,
                                   false));
  //*********** Finally, do the real work
  
  CPUTimer timer;
  timer.start();
    
  double time_of_last_stored_event = 0;
  long num_stored_events = 0;
  long num_events_in_frame = 0;
  Array<2,float> data_fan_sums(IndexRange2D(num_rings, num_detectors_per_ring));
  

  // the next variable is used to see if there are more events to store for the current segments
  // num_events_to_store-more_events will be the number of allowed coincidence events currently seen in the file
  // ('allowed' independent on the fact of we have its segment in memory or not)
  long more_events = num_events_to_store;
  
  // go to the beginning of the binary data
  {
    input.seekg(input_file_offset,ios::beg);
    
    if (input.bad()) 
      error("Problem after seeking to start of listmode file");
    
      if (input.eof()) 
	{ 
        // Strangely enough, once you read past EOF, even seekg(0) doesn't reset the eof flag
	  input.clear();
	  if (input.eof()) 
	    error("seekg forgot to reset EOF or the file is empty. Can't correct this. Exiting...\n");      
	}
  }
  
  {      
    // loop over all events in the listmode file
    CLIST_RECORD event;
    double current_time = 0;
    while (more_events)
      {
        if (!get_next_event(input, event)) 
	  {
	    // no more events in file for some reason
	    break; //get out of while loop
	  }
        if (event.is_time())
	  {
	    const double new_time = event.time.get_time_in_secs();
	    if (do_time_frame && new_time >= end_time)
	      break; // get out of while loop
	    current_time = new_time;
	  }
	else if (event.is_event() && start_time <= current_time)
	  {
            // see if we increment or decrement the value in the sinogram
            const int event_increment =
              event.event.is_prompt() 
              ? ( store_prompts ? 1 : 0 ) // it's a prompt
              :  delayed_increment;//it is a delayed-coincidence event
            
            if (event_increment==0)
              continue;
            
            more_events-= event_increment;

#ifdef HIDACREBINNER
	    if (event.event.conver_1 > max_converter ||
		event.event.conver_2 > max_converter) 
	      continue;
#error dont know how to do this
#else
            int ra,a,rb,b;
	    event.event.get_detectors(a,b,ra,rb);
#if 0
	    if (interactive)
            {
              printf("%c ra=%3d a=%4d, rb=%3d b=%4d, time=%8g",
		     event.event.is_prompt() ? 'p' : 'd',
		     ra,a,rb,b,
		     current_time);
            }
#endif
	    if (abs(ra-rb)<=max_ring_diff)
	      {
		const int det_num_diff =
		  (a-b+3*num_detectors_per_ring/2)%num_detectors_per_ring;
		if (det_num_diff<=fan_size/2 || 
		    det_num_diff>=num_detectors_per_ring-fan_size/2)
		  {
                  if (interactive)
#if 0
		      printf(" accepted\n");
#else
                  {
                    printf("%c ra=%3d a=%4d, rb=%3d b=%4d, time=%8g accepted\n",
                      event.event.is_prompt() ? 'p' : 'd',
                      ra,a,rb,b,
                      current_time);
                    
                    Bin bin;
                    proj_data_info_ptr->get_bin_for_det_pair(bin,a, ra, b, rb);
                    printf("Seg %4d view %4d ax_pos %4d tang_pos %4d\n", 
                      bin.segment_num(), bin.view_num(), bin.axial_pos_num(), bin.tangential_pos_num());
                  }
#endif
		    data_fan_sums[ra][a] += event_increment;
		    data_fan_sums[rb][b] += event_increment;
		    num_stored_events += event_increment;
		  }
		else
		  {
#if 0
                    if (interactive)		  
		      printf(" ignored\n");
#endif
		  }
	      }
	    else
	      {
#if 0
              if (interactive)		  
		  printf(" ignored\n");
#endif
	      }
	    // TODO handle do_normalisation
#endif
	    
	  } // end of spatial event processing
      } // end of while loop over all events

    time_of_last_stored_event = 
      max(time_of_last_stored_event,current_time); 
  } // end of if (num_stored_events < num_events_to_stored)


  timer.stop();
  // write fan sums to file
  {
    ofstream out(output_filename.c_str());
    out << data_fan_sums;
  }

  cerr << "Last stored event was recorded after time-tick at " << time_of_last_stored_event << " secs\n";
  if (!do_time_frame && num_stored_events<num_events_to_store) 
    cerr << "Early stop due to EOF. " << endl;
  cerr <<  "Total number of prompts/trues/delayed in this time period: " << num_events_in_frame << endl;
  cerr << "Total number of prompts/trues/delayed stored: " << num_stored_events << endl;

  cerr << "\nThis took " << timer.value() << "s CPU time." << endl;

  return EXIT_SUCCESS;
}



/************************* Local helper routines *************************/
