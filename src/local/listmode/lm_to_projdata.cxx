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
//#define INCLUDE_NORMALISATION_FACTORS

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
#include "stir/interfile.h"
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

#include "stir/CPUTimer.h"

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


#ifdef USE_SegmentByView
#include "stir/SegmentByView.h"
typedef SegmentByView<elem_type> segment_type;
#else
#include "stir/Array.h"
#include "stir/IndexRange3D.h"
typedef Array<3,elem_type> segment_type;
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

// output parameter needs to be passed because save_and_delete_segments might need it 
// (i.e. if we're not using SegmentByView)
static
shared_ptr<ProjData>
construct_proj_data_from_template(shared_ptr<iostream>& output,
                                  const string& output_filename, 
                                  const string& template_proj_data_name);

static
shared_ptr<ProjData>
construct_proj_data_by_asking(shared_ptr<iostream>& output,
                              const string& output_filename);


/************************ main ************************/


int main(int argc, char * argv[])
{
  
  if (argc<3 || argc>4) {
    cerr << "Usage: " << argv[0] << " outfilename listmode_filename  [template_projdata_filename]\n";
    exit(EXIT_FAILURE);
  }

  const string input_filename = argv[2];
  const string output_filename = argv[1];

  //*********** open listmode file


#ifdef HIDACREBINNER
  unsigned long input_file_offset = 0;
  LM_DATA_INFO lm_infos;
  read_lm_QHiDAC_data_head_only(&lm_infos,&input_file_offset,input_filename);
  shared_ptr<CListModeData> listmode_ptr =
    new CListModeDataFromStream(input_filename, input_file_offset);
#else
  // something similar will be done for other listmode types. TODO
  shared_ptr<CListModeData> listmode_ptr =
    CListModeData::read_from_file(input_filename);
#endif

  //*********** open output file

  shared_ptr<iostream> output;
  shared_ptr<ProjData> proj_data_ptr = 
    argc>3 
    ? construct_proj_data_from_template(output, output_filename, argv[3])
    : construct_proj_data_by_asking(output, output_filename);
  

  //*********** ask some more questions on how to process the data

  const bool interactive = ask("Output coords to stdout?",false);

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
    CListEvent::has_delayeds() ? ask("Store 'prompts' ?",true) : true;
  const int delayed_increment = 
    CListEvent::has_delayeds() 
    ? 
    ( store_prompts ?
      (ask("Subtract 'delayed' coincidences ?",true) ? -1 : 0)
      :
      1
    )
    :
    0 /* listmode file does not store delayeds*/; 
    


  const bool do_time_frame = ask("Do time frame",true);
  // next variable will only be used when do_time_frame==false
  unsigned long num_events_to_store = 0;
  double start_time = 0;
  double end_time = 0;

  if (do_time_frame)
  {
    start_time = ask_num("Start time (in secs)",0.,1.e6,0.);
    end_time = ask_num("End time (in secs)",start_time,1.e6,3600.); // TODO get sensible maximum from data
  }
  else
  {
    unsigned long max_num_events = 1UL << 8*sizeof(unsigned long)-1;
      //listmode_ptr->get_num_records();

    num_events_to_store = 
      ask_num("Number of (prompt/true/random) events to store", 
      (unsigned long)0, max_num_events, max_num_events);
  }
  
  const int num_segments = proj_data_ptr->get_num_segments();
  const int num_segments_in_memory = 
    ask_num("Number of segments in memory ?", 1, num_segments, num_segments);
  
  
  //*********** get proj_data_info for use in the rebinning below

#ifdef HIDACREBINNER
    const ProjDataInfoCylindrical * proj_data_info_ptr =
      dynamic_cast<const ProjDataInfoCylindrical *>
         (proj_data_ptr->get_proj_data_info_ptr());
#else
    const ProjDataInfoCylindricalNoArcCorr * proj_data_info_ptr =
      dynamic_cast<const ProjDataInfoCylindricalNoArcCorr *>
         (proj_data_ptr->get_proj_data_info_ptr());
#endif
  assert(proj_data_info_ptr != NULL);  
  

  //*********** Finally, do the real work
  
  CPUTimer timer;
  timer.start();
    
  double time_of_last_stored_event = 0;
  long num_stored_events = 0;
  long num_events_in_frame = 0;
  VectorWithOffset<segment_type *> 
    segments (proj_data_ptr->get_min_segment_num(), proj_data_ptr->get_max_segment_num());
  
  /* Here starts the main loop which will store the listmode data.
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

    // go to the beginning of the binary data
    listmode_ptr->reset();

    if (do_time_frame || 
	num_stored_events<=0 ||
	static_cast<unsigned long>(num_stored_events)<num_events_to_store) // code works without this if as well, but it might mean we're going through the listmode data without needing to
    {      
      // loop over all events in the listmode file
      CListRecord record;
      double current_time = 0;
      while (more_events)
      {
        if (listmode_ptr->get_next_record(record) == Succeeded::no) 
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
#ifdef HIDACREBINNER
          if (record.event.conver_1 > max_converter ||
            record.event.conver_2 > max_converter) // KT 03/07/2002 bug fix: was conver_1
            continue;
          
          clist_2_sinograms (bin,
            record.event,lm_infos,
            *proj_data_info_ptr,
            handle_anode_wire_efficiency);
#else
          record.event.get_bin(bin, *proj_data_info_ptr); 
          // TODO handle do_normalisation
#endif
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
                (do_normalisation ? bin.get_bin_value() : 1) *
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
    } // end of if (num_stored_events < num_events_to_store)


    save_and_delete_segments(output, segments, 
                             start_segment_index, end_segment_index, 
			     *proj_data_ptr);  
  } // end of for loop for segment range

  timer.stop();
  cerr << "Last stored event was recorded after time-tick at " << time_of_last_stored_event << " secs\n";
  if (!do_time_frame && 
      (num_stored_events<=0 ||
       static_cast<unsigned long>(num_stored_events)<num_events_to_store))
    cerr << "Early stop due to EOF. " << endl;
  cerr <<  "Total number of prompts/trues/delayed in this time period: " << num_events_in_frame << endl;
  cerr << "Total number of prompts/trues/delayed stored: " << num_stored_events << endl;

  cerr << "\nThis took " << timer.value() << "s CPU time." << endl;

  return EXIT_SUCCESS;
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

static
shared_ptr<ProjData>
construct_proj_data_from_template(shared_ptr<iostream>& output,
                                  const string& output_filename, 
                                  const string& template_proj_data_name)
{
  shared_ptr<ProjData> template_proj_data_ptr =
    ProjData::read_from_file(template_proj_data_name);

  shared_ptr<ProjDataInfo> proj_data_info_ptr = 
    template_proj_data_ptr->get_proj_data_info_ptr()->clone();
  return
    construct_proj_data(output, output_filename, proj_data_info_ptr);
}

static
shared_ptr<ProjData>
construct_proj_data_by_asking(shared_ptr<iostream>& output,
                              const string& output_filename)
{
#ifdef HIDACREBINNER
  REBINNING_INFOS rebinning_infos;
  rebinning_infos.sampling_xy = ask_num("Sampling distance (dxyr) ", 0., 4., 2.);   
  rebinning_infos.nbphi = ask_num("Number of views (nbphi) ", 0, 200, 80);
  rebinning_infos.dtheta = ask_num("Theta samples (dtheta) ", 0., _PI/2.,0.0698131700797732); 
  rebinning_infos.nbtheta = ask_num("Number of theta samples (nbtheta) ", 0, 100, 1);
  rebinning_infos.ring_radius = 1000.;  
  const int max_num_of_axial_pos = rebinning_infos.nby = ask_num("Number of axial positions ", 0, 300, 80);
  const int max_num_of_bins = ask_num("Number of tangential positions ", 0, 300, 80);

  output = new fstream (output_filename.c_str(), ios::out|ios::binary);
  if (!*output)
    error("Error opening output file %s\n",output_filename.c_str());
  // output the relevant data to the file 
  /*
   *output << "Title=Rebinned data" << endl;
   *output << "RingRadius=" << rebinning_infos.ring_radius << endl;
   *output << "dxyr=" << rebinning_infos.sampling_xy << endl;
   *output << "nby=" << rebinning_infos.nby << endl;
   *output << "dtheta=" << rebinning_infos.dtheta << endl;
   *output << "nbtheta=" << rebinning_infos.nbtheta << endl;
   *output << "nbphi="<< rebinning_infos.nbphi << endl;
   *output << "\f";  
  */
  ProjDataFromStream* proj_data_ptr = 
    create_HiDAC_PDFS(output,0L /*TODO output->tellg()*/,
		      rebinning_infos.nbphi, rebinning_infos.nbtheta,
                      rebinning_infos.dtheta, rebinning_infos.sampling_xy,
                      max_num_of_axial_pos, max_num_of_bins,false);
  write_basic_interfile_PDFS_header(output_filename, *proj_data_ptr);
  return proj_data_ptr;
#else
  shared_ptr<Scanner> scanner_ptr = new Scanner(Scanner::E966); // TODO fixed to 966 at the moment
  const int span = ask_num("span (must be odd)",1,9,1);
  const int max_delta = 
    ask_num("Max ring difference",
             0,scanner_ptr->get_num_rings(),scanner_ptr->get_num_rings());
  const int num_views =
    scanner_ptr->get_max_num_views()/ask_num("View mashing factor (must divide num_views)",1,16,1);
  const int num_tangential_poss =
    ask_num("Number of tangential positions",
            1, scanner_ptr->get_max_num_non_arccorrected_bins(), scanner_ptr->get_default_num_arccorrected_bins());


  shared_ptr<ProjDataInfo> proj_data_info_ptr =
    ProjDataInfo::ProjDataInfoCTI(scanner_ptr,
		    span, max_delta, num_views, num_tangential_poss, 
                    /*arc_corrected = */ false);
  return
    construct_proj_data(output, output_filename, proj_data_info_ptr);
#endif

}


