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
USE_SegmentByView
  Currently our ProjData classes store segments as floats, which is a waste of
  memory and time for simple binning of listmode data. This should be
  remedied at some point by having member template functions to allow different
  data types in ProjData et al.
  Currently we work (somewhat tediously) around this problem by using Array classes directly.
  If you want to use the Segment classes (safer and cleaner)
  #define USE_SegmentByView
*/   
#define USE_SegmentByView

//#define HIDACREBINNER   
#define INCLUDE_NORMALISATION_FACTORS

// set elem_type to what you want to use for the sinogram elements
// we need a signed type, as randoms can be subtracted. However, signed char could do.

#if defined(USE_SegmentByView) || defined(INCLUDE_NORMALISATION_FACTORS) 
   typedef float elem_type;
#  define OUTPUTNumericType NumericType::FLOAT
#else
   typedef short elem_type;
#  define OUTPUTNumericType NumericType::SHORT
#endif


#include "stir/utilities.h"
#include "stir/IO/interfile.h"
#include "local/stir/listmode/LmToProjData.h"
#ifdef HIDACREBINNER
#include "local/stir/QHidac/lm_qhidac.h"
#include "stir/ProjDataInfoCylindrical.h"
#else
#include "local/stir/listmode/lm.h"
#include "local/stir/listmode/CListModeData.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#endif
#include "stir/Scanner.h"
#include "stir/SegmentByView.h"
#include "stir/ProjDataFromStream.h"
#include "stir/ProjDataInterfile.h"
#include "stir/ParsingObject.h"
#include "local/stir/listmode/TimeFrameDefinitions.h"
#include "stir/CPUTimer.h"
#include "stir/recon_buildblock/TrivialBinNormalisation.h"
#include "stir/is_null_ptr.h"

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





#ifdef USE_SegmentByView
#include "stir/SegmentByView.h"
#else
#include "stir/Array.h"
#include "stir/IndexRange3D.h"
#endif

START_NAMESPACE_STIR

#ifdef USE_SegmentByView
typedef SegmentByView<elem_type> segment_type;
#else
#include "stir/Array.h"
#include "stir/IndexRange3D.h"
#endif
/******************** Prototypes  for local routines ************************/



static void 
allocate_segments(VectorWithOffset<segment_type *>& segments,
                       const int start_segment_index, 
	               const int end_segment_index,
                       const ProjDataInfo* proj_data_info_ptr);
/* last parameter only used if USE_SegmentByView
   first parameter only used when not USE_SegmentByView
 */         
static void 
save_and_delete_segments(shared_ptr<iostream>& output,
			      VectorWithOffset<segment_type *>& segments,
			      const int start_segment_index, 
			      const int end_segment_index, 
			      ProjData& proj_data);

// In the next 3 functions, the 'output' parameter needs to be passed 
// because save_and_delete_segments needs it when we're not using SegmentByView
static
shared_ptr<ProjData>
construct_proj_data(shared_ptr<iostream>& output,
                    const string& output_filename, 
                    const shared_ptr<ProjDataInfo>& proj_data_info_ptr);


void 
LmToProjData::
set_defaults()
{
  max_segment_num_to_process = -1;
  store_prompts = true;
  delayed_increment = -1;
  interactive=false;
  num_segments_in_memory = -1;
  normalisation_ptr = new TrivialBinNormalisation;
  pre_or_post_normalisation =0;
  
}

void 
LmToProjData::
initialise_keymap()
{
  parser.add_start_key("lm_to_projdata Parameters");
  parser.add_key("input file",&input_filename);
  parser.add_key("template_projdata", &template_proj_data_name);
  parser.add_key("frame_definition file",&frame_definition_filename);
  parser.add_key("output filename prefix",&output_filename_prefix);
  parser.add_parsing_key("Bin Normalisation type", &normalisation_ptr);
  parser.add_key("maximum absolute segment number to process", &max_segment_num_to_process); 
  parser.add_key("pre normalisation (1) or post_normalisation(0)", &pre_or_post_normalisation);
  parser.add_key("num_segments_in_memory", &num_segments_in_memory);

  if (CListEvent::has_delayeds())
  {
    parser.add_key("Store 'prompts'",&store_prompts);
    parser.add_key("increment to use for 'delayeds'",&delayed_increment);
  }
  parser.add_key("List event coordinates",&interactive);
  parser.add_stop_key("END");

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
  

}


bool
LmToProjData::
post_processing()
{

  if (input_filename.size()==0)
    {
      warning("You have to specify an input_filename\n");
      return true;
    }

  if (is_null_ptr(normalisation_ptr))
  {
    warning("Invalid normalisation object\n");
    return true;
  }

  
#ifdef HIDACREBINNER
  unsigned long input_file_offset = 0;
  LM_DATA_INFO lm_infos;
  read_lm_QHiDAC_data_head_only(&lm_infos,&input_file_offset,input_filename);
  lm_data_ptr =
    new CListModeDataFromStream(input_filename, input_file_offset);
#else
  // something similar will be done for other listmode types. TODO
  lm_data_ptr =
    CListModeData::read_from_file(input_filename);
#endif

  if (template_proj_data_name.size()==0)
    {
      warning("You have to specify template_projdata\n");
      return true;
    }
  shared_ptr<ProjData> template_proj_data_ptr =
    ProjData::read_from_file(template_proj_data_name);

  template_proj_data_info_ptr = 
    template_proj_data_ptr->get_proj_data_info_ptr()->clone();

  shared_ptr<Scanner> scanner_ptr = 
    new Scanner(*template_proj_data_info_ptr->get_scanner_ptr());

   proj_data_info_cyl_uncompressed_ptr =
    dynamic_cast<ProjDataInfoCylindricalNoArcCorr *>(
    ProjDataInfo::ProjDataInfoCTI(scanner_ptr, 
                  1, scanner_ptr->get_num_rings()-1,
                  scanner_ptr->get_num_detectors_per_ring()/2,
                  scanner_ptr->get_default_num_arccorrected_bins(), 
                  false));
       // set up normalisation object
  if ( normalisation_ptr->set_up(proj_data_info_cyl_uncompressed_ptr)
      != Succeeded::yes)
    error("correct_projdata: set-up of normalisation failed\n");


  if (max_segment_num_to_process==-1)
    max_segment_num_to_process = 
      template_proj_data_info_ptr->get_max_segment_num();
  else
    {
      max_segment_num_to_process =
	min(max_segment_num_to_process, 
	    template_proj_data_info_ptr->get_max_segment_num());
      template_proj_data_info_ptr->
	reduce_segment_range(-max_segment_num_to_process,
			     max_segment_num_to_process);
    }

  const int num_segments = template_proj_data_info_ptr->get_num_segments();
  if (num_segments_in_memory == -1)
    num_segments_in_memory = num_segments;
  else
    num_segments_in_memory =
      min(num_segments_in_memory, num_segments);

  frame_defs = TimeFrameDefinitions(frame_definition_filename);

  do_time_frame = true;
  // TODO handle num_events stuff

  return false;
}

LmToProjData::
LmToProjData()
{}

LmToProjData::
LmToProjData(const char * const par_filename)
{
  set_defaults();
  if (par_filename!=0)
    parse(par_filename) ;
  else
    ask_parameters();

}

void
LmToProjData::
get_bin_from_record(Bin& bin, const CListRecord& record,
		    const double time,
		    const ProjDataInfoCylindrical& proj_data_info) const
{  
#ifdef HIDACREBINNER
  if (record.event.conver_1 > max_converter ||
			 record.event.conver_2 > max_converter) // KT 03/07/2002 bug fix: was conver_1
			 continue;
  
  clist_2_sinograms (bin,
    record.event,lm_infos,
    *proj_data_info_ptr,
    handle_anode_wire_efficiency);
#else
  if (pre_or_post_normalisation)
  {
    record.event.get_bin(bin, dynamic_cast<const ProjDataInfoCylindrical&>(*proj_data_info_cyl_uncompressed_ptr)); 
    // do_normalisation
    if (bin.get_bin_value()>0)
    {
      const float bin_efficiency = normalisation_ptr->get_bin_efficiency(bin);
      bin.set_bin_value(1/bin_efficiency);
    }
    // do motion correction here

    // find detectors
  int det_num_a;
  int det_num_b;
  int ring_a;
  int ring_b;
  record.event.get_detectors(det_num_a,det_num_b,ring_a,ring_b);

  const Scanner * const scanner_ptr = 
    template_proj_data_info_ptr->get_scanner_ptr();

    if ( ring_a > scanner_ptr->get_num_rings() || ring_a <0 || ring_b <0 || 
      ring_b > scanner_ptr->get_num_rings() ||
      dynamic_cast<const ProjDataInfoCylindricalNoArcCorr&>(proj_data_info).
      get_bin_for_det_pair(bin,
      det_num_a, ring_a,
      det_num_b, ring_b) == Succeeded::no)
    {
      bin.segment_num() = 
	(ring_b-ring_a)/
	  (proj_data_info.get_max_ring_difference(0) -
	  proj_data_info.get_min_ring_difference(0) + 1);
      bin.set_bin_value(-1);

    }
  }
  else // post_normalisation
  {
    record.event.get_bin(bin, proj_data_info); 
    if (bin.get_bin_value()>0)
    {
      const float bin_efficiency = normalisation_ptr->get_bin_efficiency(bin);
      bin.set_bin_value(1/bin_efficiency);
    }
  }
#endif
}

void
LmToProjData::
compute()
{  
  //*********** get proj_data_info for use in the rebinning below

#ifdef HIDACREBINNER
  const ProjDataInfoCylindrical * proj_data_info_ptr =
    dynamic_cast<const ProjDataInfoCylindrical *>
    (template_proj_data_info_ptr.get());
#else
  const ProjDataInfoCylindricalNoArcCorr * proj_data_info_ptr =
    dynamic_cast<const ProjDataInfoCylindricalNoArcCorr *>
    (template_proj_data_info_ptr.get());
#endif
  assert(proj_data_info_ptr != NULL);  
  

  //*********** Finally, do the real work
    
    CPUTimer timer;
    timer.start();
    
    double time_of_last_stored_event = 0;
    long num_stored_events = 0;
    long num_events_in_frame = 0;
    VectorWithOffset<segment_type *> 
      segments (template_proj_data_info_ptr->get_min_segment_num(), 
		template_proj_data_info_ptr->get_max_segment_num());
    
    /* Here starts the main loop which will store the listmode data. */
 VectorWithOffset<CListModeData::SavedPosition> 
   frame_start_positions(1, frame_defs.get_num_frames());

 for (unsigned int current_frame_num = 1;
      current_frame_num<=frame_defs.get_num_frames();
      ++current_frame_num)
   {
     frame_start_positions[current_frame_num] = 
       lm_data_ptr->save_get_position();
     const double start_time = frame_defs.get_start_time(current_frame_num);
     const double end_time = frame_defs.get_end_time(current_frame_num);


     //*********** open output file
       shared_ptr<iostream> output;
       shared_ptr<ProjData> proj_data_ptr;

       {
	 char rest[50];
	 sprintf(rest, "_f%dg1b0d0", current_frame_num);
	 const string output_filename = output_filename_prefix + rest;
      
	 proj_data_ptr = 
	   construct_proj_data(output, output_filename, template_proj_data_info_ptr);
       }
       /*
	 For each start_segment_index, we check which events occur in the
	 segments between start_segment_index and 
	 start_segment_index+num_segments_in_memory.
       */
       for (int start_segment_index = proj_data_ptr->get_min_segment_num(); 
	    start_segment_index <= proj_data_ptr->get_max_segment_num(); 
	    start_segment_index += num_segments_in_memory) 
	 {
	 
	   cerr << "Processing next batch of segments" <<endl;
	   const int end_segment_index = 
	     min( proj_data_ptr->get_max_segment_num()+1, start_segment_index + num_segments_in_memory) - 1;
    
	   allocate_segments(segments, start_segment_index, end_segment_index, proj_data_ptr->get_proj_data_info_ptr());

	   // the next variable is used to see if there are more events to store for the current segments
	   // num_events_to_store-more_events will be the number of allowed coincidence events currently seen in the file
	   // ('allowed' independent on the fact of we have its segment in memory or not)
	   // When do_time_frame=true, the number of events is irrelevant, so we 
	   // just set more_events to 1, and never change it
	   long more_events = 
	     do_time_frame? 1 : num_events_to_store;

	   // go to the beginning of the listmode data for this frame
	   lm_data_ptr->set_get_position(frame_start_positions[current_frame_num]);
	   {      
	     // loop over all events in the listmode file
	     CListRecord record;

	     double current_time = start_time;
	     while (more_events)
	       {
		 if (lm_data_ptr->get_next_record(record) == Succeeded::no) 
		   {
		     // no more events in file for some reason
		     break; //get out of while loop
		   }
		 if (record.is_time())
		   {
		     const double new_time = record.time.get_time_in_secs();
		     if (do_time_frame && new_time >= end_time)
		       break; // get out of while loop
		     current_time = new_time;
		   }
		 else if (record.is_event() && start_time <= current_time)
		   {
		     Bin bin;
		     // set value in case the event decoder doesn't touch it
		     // otherwise it would be 0 and all events will be ignored
		     bin.set_bin_value(1);
                     get_bin_from_record(bin, record, current_time,*proj_data_info_ptr);
		     // check if it's inside the range we want to store
		     if (bin.get_bin_value()>0
			 && bin.tangential_pos_num()>= proj_data_ptr->get_min_tangential_pos_num()
			 && bin.tangential_pos_num()<= proj_data_ptr->get_max_tangential_pos_num()
			 && bin.axial_pos_num()>=proj_data_ptr->get_min_axial_pos_num(bin.segment_num())
			 && bin.axial_pos_num()<=proj_data_ptr->get_max_axial_pos_num(bin.segment_num())
			 ) 
		       {
			 assert(bin.view_num()>=proj_data_ptr->get_min_view_num());
			 assert(bin.view_num()<=proj_data_ptr->get_max_view_num());
            
			 // see if we increment or decrement the value in the sinogram
			 const int event_increment =
			   record.event.is_prompt() 
			   ? ( store_prompts ? 1 : 0 ) // it's a prompt
			   :  delayed_increment;//it is a delayed-coincidence event
            
			 if (event_increment==0)
			   continue;
            
			 if (!do_time_frame)
			   more_events-= event_increment;
            
			 // now check if we have its segment in memory
			 if (bin.segment_num() >= start_segment_index && bin.segment_num()<=end_segment_index)
			   {
			     num_events_in_frame += event_increment; 
			     if (interactive)
			       printf("Seg %4d view %4d ax_pos %4d tang_pos %4d time %8g stored\n", 
				      bin.segment_num(), bin.view_num(), bin.axial_pos_num(), bin.tangential_pos_num(),
				      current_time);
              
			     num_stored_events += event_increment;
			     if (num_stored_events%500000L==0) cout << "\r" << num_stored_events << flush;
                            
			     (*segments[bin.segment_num()])[bin.view_num()][bin.axial_pos_num()][bin.tangential_pos_num()] += 
#ifdef INCLUDE_NORMALISATION_FACTORS
			       bin.get_bin_value() * // TODO HIDAC
#endif
			       event_increment;
			   }
		       }
		     else 	// event is rejected for some reason
		       {
			 // we could just do nothing here if we didn't report 
			 // num_events_in_frame nor had the 'interactive' option
            
			 if (bin.segment_num() >= start_segment_index && bin.segment_num()<=end_segment_index)
			   {
			     const int event_increment =
			       record.event.is_prompt() 
			       ? ( store_prompts ? 1 : 0 ) // it's a prompt
			       :  delayed_increment;//it is a delayed-coincidence event
			     if (!event_increment)
			       continue;
              
			     num_events_in_frame += event_increment; 
			     if (interactive)
			       printf("Seg %4d view %4d ax_pos %4d tang_pos %4d time %8g ignored\n", 
				      bin.segment_num(), bin.view_num(), bin.axial_pos_num(), bin.tangential_pos_num(), current_time);
			   }
		       }     
		   } // end of spatial event processing
	       } // end of while loop over all events

	     time_of_last_stored_event = 
	       max(time_of_last_stored_event,current_time); 
	   } 


	   save_and_delete_segments(output, segments, 
				    start_segment_index, end_segment_index, 
				    *proj_data_ptr);  
	 } // end of for loop for segment range
   } // end of loop over frames

 timer.stop();

 cerr << "Last stored event was recorded after time-tick at " << time_of_last_stored_event << " secs\n";
 if (!do_time_frame && 
     (num_stored_events<=0 ||
      static_cast<unsigned long>(num_stored_events)<num_events_to_store))
   cerr << "Early stop due to EOF. " << endl;
 cerr <<  "Total number of prompts/trues/delayed within segment limit in this time period: " << num_events_in_frame << endl;
 cerr << "Total number of prompts/trues/delayed stored: " << num_stored_events << endl;

 cerr << "\nThis took " << timer.value() << "s CPU time." << endl;

}



/************************* Local helper routines *************************/


void 
allocate_segments( VectorWithOffset<segment_type *>& segments,
		  const int start_segment_index, 
		  const int end_segment_index,
		  const ProjDataInfo* proj_data_info_ptr)
{
  
  for (int seg=start_segment_index ; seg<=end_segment_index; seg++)
  {
#ifdef USE_SegmentByView
    segments[seg] = new SegmentByView<elem_type>(
    	proj_data_info_ptr->get_empty_segment_by_view (seg)); 
#else
    segments[seg] = 
      new Array<3,elem_type>(IndexRange3D(0, proj_data_info_ptr->get_num_views()-1, 
				      0, proj_data_info_ptr->get_num_axial_poss(seg)-1,
				      -(proj_data_info_ptr->get_num_tangential_poss()/2), 
				      proj_data_info_ptr->get_num_tangential_poss()-(proj_data_info_ptr->get_num_tangential_poss()/2)-1));
#endif
  }
}

void 
save_and_delete_segments(shared_ptr<iostream>& output,
			 VectorWithOffset<segment_type *>& segments,
			 const int start_segment_index, 
			 const int end_segment_index, 
			 ProjData& proj_data)
{
  
  for (int seg=start_segment_index; seg<=end_segment_index; seg++)
  {
    {
#ifdef USE_SegmentByView
      proj_data.set_segment(*segments[seg]);
#else
      (*segments[seg]).write_data(*output);
#endif
      delete segments[seg];      
    }
    
  }
}



static
shared_ptr<ProjData>
construct_proj_data(shared_ptr<iostream>& output,
                    const string& output_filename, 
                    const shared_ptr<ProjDataInfo>& proj_data_info_ptr)
{
  vector<int> segment_sequence_in_stream(proj_data_info_ptr->get_num_segments());
  { 
#ifndef STIR_NO_NAMESPACES
    std:: // explcitly needed by VC
#endif
    vector<int>::iterator current_segment_iter =
      segment_sequence_in_stream.begin();
    for (int segment_num=proj_data_info_ptr->get_min_segment_num();
         segment_num<=proj_data_info_ptr->get_max_segment_num();
         ++segment_num)
      *current_segment_iter++ = segment_num;
  }
#ifdef USE_SegmentByView
  // don't need output stream in this case
  return new ProjDataInterfile(proj_data_info_ptr, output_filename, ios::out, 
                               segment_sequence_in_stream,
                               ProjDataFromStream::Segment_View_AxialPos_TangPos,
		               OUTPUTNumericType);
#else
  // this code would work for USE_SegmentByView as well, but the above is far simpler...
  output = new fstream (output_filename.c_str(), ios::out|ios::binary);
  if (!*output)
    error("Error opening output file %s\n",output_filename.c_str());
  shared_ptr<ProjDataFromStream> proj_data_ptr = 
    new ProjDataFromStream(proj_data_info_ptr, output, 
                           /*offset=*/0, 
                           segment_sequence_in_stream,
                           ProjDataFromStream::Segment_View_AxialPos_TangPos,
		           OUTPUTNumericType);
  write_basic_interfile_PDFS_header(output_filename, *proj_data_ptr);
  return proj_data_ptr;  
#endif
}


END_NAMESPACE_STIR
