/*!
  \file 
  \ingroup listmode

  \brief Implementation of class stir::LmToProjData
 
  \author Nikos Efthimiou
  \author Kris Thielemans
  \author Sanida Mustafovic
*/
/*
    Copyright (C) 2000 - 2011-12-31, Hammersmith Imanet Ltd
    Copyright (C) 2017, University of Hull
    Copyright (C) 2013, 2021 University College London
    Copright (C) 2019, National Physical Laboratory
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

/* Possible compilation switches:
  
USE_SegmentByView 
  Currently our ProjData classes store segments as floats, which is a waste of
  memory and time for simple binning of listmode data. This should be
  remedied at some point by having member template functions to allow different
  data types in ProjData et al.
  Currently we work (somewhat tediously) around this problem by using Array classes directly.
  If you want to use the Segment classes (safer and cleaner)
  #define USE_SegmentByView


FRAME_BASED_DT_CORR:
   dead-time correction based on the frame, or on the time of the event
*/   
// (Note: can currently NOT be disabled)
#define USE_SegmentByView

//#define FRAME_BASED_DT_CORR

// set elem_type to what you want to use for the sinogram elements
// we need a signed type, as randoms can be subtracted. However, signed char could do.

#if defined(USE_SegmentByView) 
   typedef float elem_type;
#  define OUTPUTNumericType NumericType::FLOAT
#else
   #error currently problem with normalisation code!
   typedef short elem_type;
#  define OUTPUTNumericType NumericType::SHORT
#endif


#include "stir/utilities.h"

#include "stir/listmode/LmToProjData.h"
#include "stir/listmode/ListRecord.h"
#include "stir/listmode/ListModeData.h"
#include "stir/ExamInfo.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"

#include "stir/Scanner.h"
#ifdef USE_SegmentByView
#include "stir/ProjDataInterfile.h"
#include "stir/SegmentByView.h"
#else
#include "stir/ProjDataFromStream.h"
#include "stir/IO/interfile.h"
#include "stir/Array.h"
#include "stir/IndexRange3D.h"
#endif
#include "stir/IO/read_from_file.h"
#include "stir/ParsingObject.h"
#include "stir/TimeFrameDefinitions.h"
#include "stir/CPUTimer.h"
#include "stir/recon_buildblock/TrivialBinNormalisation.h"
#include "stir/is_null_ptr.h"

#include <fstream>
#include <iostream>
#include <vector>

#ifndef STIR_NO_NAMESPACES
using std::string;
using std::fstream;
using std::ifstream;
using std::iostream;
using std::ofstream;
using std::ios;
using std::cerr;
using std::cout;
using std::flush;
using std::endl;
using std::min;
using std::max;
using std::vector;
using std::pair;
#endif

START_NAMESPACE_STIR

#ifdef USE_SegmentByView
typedef SegmentByView<elem_type> segment_type;
#else
#error does not work at the moment
#endif
/******************** Prototypes  for local routines ************************/



static void 
allocate_segments(VectorWithOffset<VectorWithOffset<segment_type *> >& segments,
		  const int start_timing_pos_index,
		  const int end_timing_pos_index,
		  const int start_segment_index,
		  const int end_segment_index,
		  const shared_ptr<const ProjDataInfo> proj_data_info_sptr);

// In the next 2 functions, the 'output' parameter needs to be passed 
// because save_and_delete_segments needs it when we're not using SegmentByView

/* last parameter only used if USE_SegmentByView
   first parameter only used when not USE_SegmentByView
 */         
static void 
save_and_delete_segments(shared_ptr<iostream>& output,
			 VectorWithOffset<VectorWithOffset<segment_type *> >& segments,
		  const int start_timing_pos_index,
		  const int end_timing_pos_index,
                  const int start_segment_index,
                  const int end_segment_index,
                  ProjData& proj_data);
static
shared_ptr<ProjData>
construct_proj_data(shared_ptr<iostream>& output,
                    const string& output_filename, 
		    const ExamInfo& exam_info,
                    const shared_ptr<const ProjDataInfo>& proj_data_info_ptr);

/**************************************************************
 set/get
**************************************************************/
void LmToProjData::set_template_proj_data_info_sptr(shared_ptr<const ProjDataInfo> t_sptr)
{
  this->_already_setup = false;
  template_proj_data_info_ptr = t_sptr->create_shared_clone();
}

shared_ptr<ProjDataInfo> LmToProjData::get_template_proj_data_info_sptr()
{
  return template_proj_data_info_ptr;
}

void LmToProjData::set_input_data(const shared_ptr<ExamData>& v)
{
  this->_already_setup = false;
  this->lm_data_ptr = dynamic_pointer_cast<ListModeData>(v);
  if (is_null_ptr(this->lm_data_ptr))
    error("LmToProjData::set_input_data() called with non-listmode data or other error");
}

void LmToProjData::set_input_data(const std::string& filename)
{
  shared_ptr<ListModeData> lm(stir::read_from_file<ListModeData>(filename));
  this->set_input_data(lm);
  this->input_filename = filename;
}

#if 0
ListModeData& LmToProjData::get_input_data()
{
  return *lm_data_ptr;
}
#endif


void LmToProjData::set_output_filename_prefix(const std::string& v)
{
  this->output_filename_prefix = v;
}

std::string LmToProjData::get_output_filename_prefix() const
{
  return output_filename_prefix;
}

void LmToProjData::set_output_projdata_sptr(shared_ptr<ProjData>& arg)
{
    this->output_proj_data_sptr = arg;
}

void LmToProjData::set_store_prompts(bool v)
{
  this->store_prompts = v;
}

bool LmToProjData::get_store_prompts() const
{
  return store_prompts;
}

void LmToProjData::set_store_delayeds(bool v)
{
  this->store_delayeds = v;
}

bool LmToProjData::get_store_delayeds() const
{
  return store_delayeds;
}

void LmToProjData::set_num_segments_in_memory(int v)
{
  this->_already_setup = false;
  this->num_segments_in_memory = v;
}

int LmToProjData::get_num_segments_in_memory() const
{
  return num_segments_in_memory;
}

void LmToProjData::set_num_events_to_store(long int v)
{
  this->num_events_to_store = v;
}

long int LmToProjData::get_num_events_to_store() const
{
  return num_events_to_store;
}

void LmToProjData::set_time_frame_definitions(const TimeFrameDefinitions& v)
{
  this->frame_defs = v;
}

const TimeFrameDefinitions& LmToProjData::get_time_frame_definitions() const
{
  return frame_defs;
}

/**************************************************************
 The 3 parsing functions
***************************************************************/
void 
LmToProjData::
set_defaults()
{
  max_segment_num_to_process = -1;
  store_prompts = true;
  store_delayeds = true;
  interactive=false;
  num_segments_in_memory = -1;
  num_timing_poss_in_memory = -1;
  normalisation_ptr.reset(new TrivialBinNormalisation);
  post_normalisation_ptr.reset(new TrivialBinNormalisation);
  do_pre_normalisation =0;
  num_events_to_store = 0L;
  do_time_frame = false;
}

void 
LmToProjData::
initialise_keymap()
{
  parser.add_start_key("lm_to_projdata Parameters");
  parser.add_key("input file",&input_filename);
  parser.add_key("template_projdata", &template_proj_data_name);
  parser.add_key("frame_definition file",&frame_definition_filename);
  parser.add_key("num_events_to_store",&num_events_to_store);
  parser.add_key("output filename prefix",&output_filename_prefix);
  parser.add_parsing_key("Bin Normalisation type for pre-normalisation", &normalisation_ptr);
  parser.add_parsing_key("Bin Normalisation type for post-normalisation", &post_normalisation_ptr);
  parser.add_key("maximum absolute segment number to process", &max_segment_num_to_process); 
  parser.add_key("do pre normalisation ", &do_pre_normalisation);
  parser.add_key("num_TOF_bins_in_memory", &num_timing_poss_in_memory);
  parser.add_key("num_segments_in_memory", &num_segments_in_memory);

  //if (lm_data_ptr->has_delayeds()) TODO we haven't read the ListModeData yet, so cannot access has_delayeds() yet
  // one could add the next 2 keywords as part of a callback function for the 'input file' keyword.
  // That's a bit too much trouble for now though...
  {
    parser.add_key("Store prompts",&store_prompts);
    parser.add_key("Store delayeds",&store_delayeds);
    //parser.add_key("increment to use for 'delayeds'",&delayed_increment);
  }
  parser.add_key("List event coordinates",&interactive);
  parser.add_stop_key("END");  

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

  set_input_data(input_filename);

  if (template_proj_data_name.size()==0)
    {
      warning("You have to specify template_projdata\n");
      return true;
    }
  shared_ptr<ProjData> template_proj_data_sptr =
    ProjData::read_from_file(template_proj_data_name);

  set_template_proj_data_info_sptr(template_proj_data_sptr->get_proj_data_info_sptr());

  if (set_up() == Succeeded::no)
    return true;

#ifdef FRAME_BASED_DT_CORR
  cerr << "LmToProjData Using FRAME_BASED_DT_CORR\n";
#else
  cerr << "LmToProjData NOT Using FRAME_BASED_DT_CORR\n";
#endif

  return false;
}

Succeeded LmToProjData::set_up()
{
  _already_setup = true;

  if (!interactive && output_filename_prefix.size()==0)
    {
      error("You have to specify an output_filename_prefix");
    }

  if (is_null_ptr(template_proj_data_info_ptr))
    {
      error("LmToProjData::set_up(): template projection data not set");
    }

  // set up normalisation objects

  if (is_null_ptr(normalisation_ptr))
    {
      error("Invalid pre-normalisation object");
    }
  if (is_null_ptr(post_normalisation_ptr))
    {
      error("Invalid post-normalisation object");
    }

  // initialise segment_num related variables

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
  if (num_segments_in_memory == -1 || interactive)
    num_segments_in_memory = num_segments;
  else
    num_segments_in_memory =
      min(num_segments_in_memory, num_segments);
  if (num_segments == 0)
    {
      error("LmToProjData: num_segments_in_memory cannot be 0");
    }
  const int num_timing_poss = template_proj_data_info_ptr->get_num_tof_poss();
  if (num_timing_poss_in_memory == -1 || interactive)
    num_timing_poss_in_memory = num_timing_poss;
  else
    num_timing_poss_in_memory =
      min(num_timing_poss_in_memory, num_timing_poss);
  if (num_timing_poss == 0)
    {
      error("LmToProjData: num_timing_poss_in_memory cannot be 0");
    }

  // handle store_prompts and store_delayeds
  
  if (store_prompts)
    {
      if (store_delayeds)
	delayed_increment = -1;
      else
	delayed_increment = 0;
    }
  else
    {
      if (store_delayeds)
	delayed_increment = 1;
      else
	{
	  error("At least one of store_prompts or store_delayeds should be true");
	}
    }

  if (do_pre_normalisation)
    {
      shared_ptr<Scanner> scanner_sptr(new Scanner(*template_proj_data_info_ptr->get_scanner_sptr()));
      // TODO this won't work for the HiDAC or so
      // N.E: The following command used to do a dynamic cast which now I removed.
      proj_data_info_cyl_uncompressed_ptr.reset(ProjDataInfo::ProjDataInfoCTI(scanner_sptr,
                                                 1, scanner_sptr->get_num_rings()-1,
                                                 scanner_sptr->get_num_detectors_per_ring()/2,
                                                 scanner_sptr->get_max_num_non_arccorrected_bins(),
                                                 false,
                                                 1));
      
      if ( normalisation_ptr->set_up(lm_data_ptr->get_exam_info_sptr(), proj_data_info_cyl_uncompressed_ptr)
	   != Succeeded::yes)
	error("LmToProjData: set-up of pre-normalisation failed\n");
    }
  else
    {
      auto all_frames_exam_info_sptr = std::make_shared<ExamInfo>(lm_data_ptr->get_exam_info());
      all_frames_exam_info_sptr->set_time_frame_definitions(frame_defs);
      if ( post_normalisation_ptr->set_up(lm_data_ptr->get_exam_info_sptr(),template_proj_data_info_ptr)
	   != Succeeded::yes)
	error("LmToProjData: set-up of post-normalisation failed\n");
    }

  // handle time frame definitions etc
  // If num_events_to_store == 0 && frame_definition_filename.size == 0
  if(num_events_to_store==0 && frame_definition_filename.size() == 0)
        do_time_frame = true;

  if (frame_definition_filename.size()!=0)
  {
    frame_defs = TimeFrameDefinitions(frame_definition_filename);
    do_time_frame = true;
  }
  else if (frame_defs.get_num_frames() < 1)
    {
      // make a single frame starting from 0. End value will be ignored.
      vector<pair<double, double> > frame_times(1, pair<double,double>(0,0));
      frame_defs = TimeFrameDefinitions(frame_times);
    }

  return Succeeded::yes;
}

/**************************************************************
 Constructors
***************************************************************/

LmToProjData::
LmToProjData()
{
  set_defaults();
}

LmToProjData::
LmToProjData(const char * const par_filename)
{
  set_defaults();
  if (par_filename!=0)
    {
      if (parse(par_filename)==false)
	error("Exiting\n");
    }
  else
    ask_parameters();
}

/**************************************************************
 Here follows the implementation of get_bin_from_event

 this function is complicated because of the normalisation stuff. sorry
***************************************************************/
void
LmToProjData::
get_bin_from_event(Bin& bin, const ListEvent& event) const
{  
  if (do_pre_normalisation)
   {
     Bin uncompressed_bin;
     event.get_bin(uncompressed_bin, *proj_data_info_cyl_uncompressed_ptr);
     if (uncompressed_bin.get_bin_value()<=0)
      return; // rejected for some strange reason


    // do_normalisation
//#ifndef FRAME_BASED_DT_CORR
//     const double start_time = current_time;
//     const double end_time = current_time;
//#else
//     const double start_time = frame_defs.get_start_time(current_frame_num);
//     const double end_time =frame_defs.get_end_time(current_frame_num);
//#endif
     
      const float bin_efficiency = 
	normalisation_ptr->get_bin_efficiency(uncompressed_bin);
      // TODO remove arbitrary number. Supposes that these bin_efficiencies are around 1
      if (bin_efficiency < 1.E-10)
	{
	warning("\nBin_efficiency %g too low for uncompressed bin (s:%d,v:%d,ax_pos:%d,tang_pos:%d). Event ignored\n",
		bin_efficiency,
		uncompressed_bin.segment_num(), uncompressed_bin.view_num(), 
		uncompressed_bin.axial_pos_num(), uncompressed_bin.tangential_pos_num());
        bin.set_bin_value(-1.f);
	return;
	}
     
    // now find 'compressed' bin, i.e. taking mashing, span etc into account
    // Also, adjust the normalisation factor according to the number of
    // uncompressed bins in a compressed bin

    const float bin_value = 1.f/bin_efficiency;
    // TODO wasteful: we decode the event twice. replace by something like
    // template_proj_data_info_ptr->get_bin_from_uncompressed(bin, uncompressed_bin);
    event.get_bin(bin, *template_proj_data_info_ptr);
   
    if (bin.get_bin_value()>0)
      {
	bin.set_bin_value(bin_value);
      }

  }
  else
    {
      event.get_bin(bin, *template_proj_data_info_ptr);
    }

} 

/**************************************************************
 Here follows the post_normalisation related stuff. 
***************************************************************/
void 
LmToProjData::
do_post_normalisation(Bin& bin) const
{
    if (bin.get_bin_value()>0)
    {
      if (do_pre_normalisation)
	{
	  bin.set_bin_value(bin.get_bin_value()/get_compression_count(bin));
	}
      else
	{
//#ifndef FRAME_BASED_DT_CORR
//	  const double start_time = current_time;
//	  const double end_time = current_time;
//#else
//	  const double start_time = frame_defs.get_start_time(current_frame_num);
//	  const double end_time =frame_defs.get_end_time(current_frame_num);
//#endif
	  const float bin_efficiency = post_normalisation_ptr->get_bin_efficiency(bin);
	  // TODO remove arbitrary number. Supposes that these bin_efficiencies are around 1
	  if (bin_efficiency < 1.E-10)
	    {
	      warning("\nBin_efficiency %g too low for bin (s:%d,v:%d,ax_pos:%d,tang_pos:%d). Event ignored\n",
		      bin_efficiency,
		      bin.segment_num(), bin.view_num(), bin.axial_pos_num(), bin.tangential_pos_num());
	      bin.set_bin_value(-1);
	    }
	  else
	    {
          bin.set_bin_value(bin.get_bin_value()/bin_efficiency);
	    }	  
	}
    }
}


int
LmToProjData::
get_compression_count(const Bin& bin) const
{
  // TODO this currently works ONLY for cylindrical PET scanners

   const ProjDataInfoCylindrical& proj_data_info =
      dynamic_cast<const ProjDataInfoCylindrical&>(*template_proj_data_info_ptr);

    return static_cast<int>(proj_data_info.get_num_ring_pairs_for_segment_axial_pos_num(bin.segment_num(),bin.axial_pos_num()))*
			   proj_data_info.get_view_mashing_factor();

}

/**************************************************************
 Empty functions for new time events and new time frames.
***************************************************************/
void
LmToProjData::
process_new_time_event(const ListTime&)
{}


void
LmToProjData::
start_new_time_frame(const unsigned int)
{}

/**************************************************************
 Here follows the actual rebinning code (finally).

 It's essentially simple, but is in fact complicated because of the facility
 to store only part of the segments in memory.
***************************************************************/
void
LmToProjData::
process_data()
{
  if (!_already_setup)
    error("LmToProjData: you need to call set_up() first");

  CPUTimer timer;
  timer.start();

  bool writing_to_file = false;

  // propagate relevant metadata
  template_proj_data_info_ptr->set_bed_position_horizontal
    (lm_data_ptr->get_proj_data_info_sptr()->get_bed_position_horizontal());
  template_proj_data_info_ptr->set_bed_position_vertical
    (lm_data_ptr->get_proj_data_info_sptr()->get_bed_position_vertical());

  // a few more checks, now that we have the lm_data_ptr
  {
    Scanner const * const scanner_ptr = 
      template_proj_data_info_ptr->get_scanner_ptr();

    if (*scanner_ptr != *lm_data_ptr->get_scanner_ptr())
      {
        error("LmToProjData:\nScanner from list mode data (%s) is different from\n"
                "scanner from template projdata (%s).\n"
                "Full definition of scanner from list mode data:\n%s\n"
                "Full definition of scanner from template:\n%s\n",
                lm_data_ptr->get_scanner_ptr()->get_name().c_str(),
                scanner_ptr->get_name().c_str(),
                lm_data_ptr->get_scanner_ptr()->parameter_info().c_str(),
                scanner_ptr->parameter_info().c_str());
      }

    if (lm_data_ptr->has_delayeds()==false && store_delayeds==true)
      {
        warning("This list mode data does not seem to have delayed events.\n"
                "Setting store_delayeds to false.");
        store_delayeds=false;
      }
  }
  // assume list mode data starts at time 0
  // we have to do this because the first time tag might occur only after a
  // few coincidence events (as happens with ECAT scanners)
  current_time = 0;

  double time_of_last_stored_event = 0;
  long num_stored_events = 0;
  VectorWithOffset<segment_type *> 
    segments (template_proj_data_info_ptr->get_min_segment_num(), 
	      template_proj_data_info_ptr->get_max_segment_num());
  
  VectorWithOffset<ListModeData::SavedPosition>
    frame_start_positions(1, static_cast<int>(frame_defs.get_num_frames()));
  shared_ptr <ListRecord> record_sptr = lm_data_ptr->get_empty_record_sptr();
  ListRecord& record = *record_sptr;

  if (!record.event().is_valid_template(*template_proj_data_info_ptr))
    error("The scanner template is not valid for LmToProjData. This might be because of unsupported arc correction.");


  /* Here starts the main loop which will store the listmode data. */
  for (current_frame_num = 1;
       current_frame_num<=frame_defs.get_num_frames();
       ++current_frame_num)
    {
      start_new_time_frame(current_frame_num);

      // construct ExamInfo appropriate for a single projdata with this time frame
      ExamInfo this_frame_exam_info(lm_data_ptr->get_exam_info());
      {
        TimeFrameDefinitions this_time_frame_defs(frame_defs, current_frame_num);
        this_frame_exam_info.set_time_frame_definitions(this_time_frame_defs);
      }

      // *********** open output file
      shared_ptr<iostream> output;
      if (!output_proj_data_sptr)
      {
        writing_to_file = true;
        char rest[50];
        sprintf(rest, "_f%dg1d0b0", current_frame_num);
        const string output_filename = output_filename_prefix + rest;
      
        output_proj_data_sptr =
          construct_proj_data(output, output_filename, this_frame_exam_info, template_proj_data_info_ptr);
      }

      long num_prompts_in_frame = 0;
      long num_delayeds_in_frame = 0;

      const double start_time = frame_defs.get_start_time(current_frame_num);
      const double end_time = frame_defs.get_end_time(current_frame_num);

      VectorWithOffset<VectorWithOffset<segment_type *> >
        segments (template_proj_data_info_ptr->get_min_tof_pos_num(),
                  template_proj_data_info_ptr->get_max_tof_pos_num());
      for (int timing_pos_num=segments.get_min_index(); timing_pos_num<=segments.get_max_index(); ++timing_pos_num)
        {
          segments[timing_pos_num].resize(template_proj_data_info_ptr->get_min_segment_num(), 
                                          template_proj_data_info_ptr->get_max_segment_num());
        }
      for (int start_timing_pos_index = output_proj_data_sptr->get_min_tof_pos_num();
           start_timing_pos_index <= output_proj_data_sptr->get_max_tof_pos_num();
           start_timing_pos_index += num_timing_poss_in_memory)
        {
	  const int end_timing_pos_index =
	    min( output_proj_data_sptr->get_max_tof_pos_num()+1,
		 start_timing_pos_index + num_timing_poss_in_memory) - 1;

	  /*
	    For each start_segment_index, we check which events occur in the
	    segments between start_segment_index and
	    start_segment_index+num_segments_in_memory.
	  */
          for (int start_segment_index = output_proj_data_sptr->get_min_segment_num();
               start_segment_index <= output_proj_data_sptr->get_max_segment_num();
               start_segment_index += num_segments_in_memory)
            {

              const int end_segment_index =
                min( output_proj_data_sptr->get_max_segment_num()+1, start_segment_index + num_segments_in_memory) - 1;

              if (!interactive)
                allocate_segments(segments,
                                  start_timing_pos_index,end_timing_pos_index,
                                  start_segment_index, end_segment_index,		   
                                  output_proj_data_sptr->get_proj_data_info_sptr());

              // the next variable is used to see if there are more events to store for the current segments
              // num_events_to_store-more_events will be the number of allowed coincidence events currently seen in the file
              // ('allowed' independent on the fact of we have its segment in memory or not)
              // When do_time_frame=true, the number of events is irrelevant, so we
              // just set more_events to 1, and never change it
              unsigned long int more_events =
                do_time_frame? 1 : num_events_to_store;

              if (start_segment_index != output_proj_data_sptr->get_min_segment_num() || start_timing_pos_index > output_proj_data_sptr->get_min_tof_pos_num())
                {
                  // we're going once more through the data (for the next batch of segments)
		  cerr << "\nProcessing next batch of segments for start TOF bin " << start_timing_pos_index <<"\n";
                  // go to the beginning of the listmode data for this frame
                  lm_data_ptr->set_get_position(frame_start_positions[current_frame_num]);
                  current_time = start_time;
                }
              else
                {
                  cerr << "\nProcessing time frame " << current_frame_num << '\n';

                  // Note: we already have current_time from previous frame, so don't
                  // need to set it. In fact, setting it to start_time would be wrong
                  // as we first might have to skip some events before we get to start_time.
                  // So, let's do that now.
                  while (current_time < start_time &&
                         lm_data_ptr->get_next_record(record) == Succeeded::yes)
                    {
                      if (record.is_time())
                        current_time = record.time().get_time_in_secs();
                    }
                  // now save position such that we can go back
                  frame_start_positions[current_frame_num] =
                    lm_data_ptr->save_get_position();
                }
              {
                // loop over all events in the listmode file
                while (more_events)
                  {
                    if (lm_data_ptr->get_next_record(record) == Succeeded::no)
                      {
                        // no more events in file for some reason
                        break; //get out of while loop
                      }
                    if (record.is_time() && end_time > 0.01) // Direct comparison within doubles is unsafe.
                      {
                        current_time = record.time().get_time_in_secs();
                        if (do_time_frame && current_time >= end_time)
                          break; // get out of while loop
                        assert(current_time>=start_time);
                        process_new_time_event(record.time());
                      }
                    // note: could do "else if" here if we would be sure that
                    // a record can never be both timing and coincidence event
                    // and there might be a scanner around that has them both combined.
                    if (record.is_event())
                      {
                        assert(start_time <= current_time);
                        Bin bin;
                        // set value in case the event decoder doesn't touch it
                        // otherwise it would be 0 and all events will be ignored
                        bin.set_bin_value(1.f);
                        bin.time_frame_num() = current_frame_num;

                        try {
                          get_bin_from_event(bin, record.event());
                        }
                        catch (...) {
                          for (int timing_pos_num=start_timing_pos_index ; timing_pos_num<=end_timing_pos_index; timing_pos_num++)
                            for (int seg=start_segment_index; seg<=end_segment_index; seg++)
                              delete segments[timing_pos_num][seg];    
                          error("Something wrong with geometry.");
                        }

                        // check if it's inside the range we want to store
                        if (bin.get_bin_value()>0
                            && bin.tangential_pos_num()>= output_proj_data_sptr->get_min_tangential_pos_num()
                            && bin.tangential_pos_num()<= output_proj_data_sptr->get_max_tangential_pos_num()
                            && bin.axial_pos_num()>=output_proj_data_sptr->get_min_axial_pos_num(bin.segment_num())
                            && bin.axial_pos_num()<=output_proj_data_sptr->get_max_axial_pos_num(bin.segment_num())
                            && bin.timing_pos_num()>=output_proj_data_sptr->get_min_tof_pos_num()
                            && bin.timing_pos_num()<=output_proj_data_sptr->get_max_tof_pos_num()
                            )
                          {
                            assert(bin.view_num()>=output_proj_data_sptr->get_min_view_num());
                            assert(bin.view_num()<=output_proj_data_sptr->get_max_view_num());

                            // see if we increment or decrement the value in the sinogram
                            const int event_increment =
                              record.event().is_prompt()
                              ? ( store_prompts ? 1 : 0 ) // it's a prompt
                              :  delayed_increment;//it is a delayed-coincidence event

                            if (event_increment==0)
                              continue;

                            if (!do_time_frame)
                              more_events -= event_increment;

                            // Check if the timing position of the bin is in the range
                            if (bin.timing_pos_num() >= start_timing_pos_index && bin.timing_pos_num()<=end_timing_pos_index)
                              {
                                // now check if we have its segment in memory
                                if (bin.segment_num() >= start_segment_index && bin.segment_num()<=end_segment_index)
                                  {
                                    do_post_normalisation(bin);

                                    num_stored_events += event_increment;
                                    if (record.event().is_prompt())
                                      ++num_prompts_in_frame;
                                    else
                                      ++num_delayeds_in_frame;

                                    if (num_stored_events%500000L==0) cout << "\r" << num_stored_events << " events stored" << flush;

                                    if (interactive)
                                      printf("TOFbin %4d Seg %4d view %4d ax_pos %4d tang_pos %4d time %8g stored with incr %d \n",
                                             bin.timing_pos_num(),bin.segment_num(),
                                             bin.view_num(), bin.axial_pos_num(), bin.tangential_pos_num(),
                                             current_time, event_increment);
                                    else
                                      (*segments[bin.timing_pos_num()][bin.segment_num()])[bin.view_num()][bin.axial_pos_num()][bin.tangential_pos_num()] +=
                                        bin.get_bin_value() *
                                        event_increment;
                                  }
                              }
                          }
                        else 	// event is rejected for some reason
                          {
                            if (interactive)
                              printf("TOFbin %4d Seg %4d view %4d ax_pos %4d tang_pos %4d time %8g ignored\n",
                                     bin.timing_pos_num(), bin.segment_num(), bin.view_num(),
                                     bin.axial_pos_num(), bin.tangential_pos_num(), current_time);
                          }
                      } // end of spatial event processing
                  } // end of while loop over all events

                time_of_last_stored_event =
                  max(time_of_last_stored_event,current_time);
              }

              if (!interactive)
                save_and_delete_segments(output, segments,
                                         start_timing_pos_index,end_timing_pos_index,
                                         start_segment_index, end_segment_index,
                                         *output_proj_data_sptr);
            } // end of for loop for segment range

        } // end of for loop for timing positions
      cerr <<  "\nNumber of prompts stored in this time period : " << num_prompts_in_frame
           <<  "\nNumber of delayeds stored in this time period: " << num_delayeds_in_frame
           << '\n';

      // if we used the member variable for writing to file, reset it to null again
      if (writing_to_file)
        {
          output_proj_data_sptr.reset();
        }
    } // end of loop over frames

    timer.stop();

    cerr << "Last stored event was recorded before time-tick at " << time_of_last_stored_event << " secs\n";
    if (!do_time_frame &&
            (num_stored_events<=0 ||
             /*static_cast<unsigned long>*/(num_stored_events)<num_events_to_store))
        cerr << "Early stop due to EOF. " << endl;
    cerr << "Total number of counts (either prompts/trues/delayeds) stored: " << num_stored_events << endl;

    cerr << "\nThis took " << timer.value() << "s CPU time." << endl;
}

void
LmToProjData::run_tof_test_function()
{
#if 1
  error("TOF test function disabled");
#else

  VectorWithOffset<VectorWithOffset<segment_type *> >
    segments (template_proj_data_info_ptr->get_min_tof_pos_num(),
              template_proj_data_info_ptr->get_max_tof_pos_num());
  for (int timing_pos_num=segments.get_min_index(); timing_pos_num<=segments.get_max_index(); ++timing_pos_num)
    {
      segments[timing_pos_num].resize(template_proj_data_info_ptr->get_min_segment_num(), 
                                      template_proj_data_info_ptr->get_max_segment_num());
    }
 
    // *********** open output file
    shared_ptr<iostream> output;
    shared_ptr<ProjData> proj_data_sptr;

    ExamInfo this_frame_exam_info(*lm_data_ptr->get_exam_info_sptr());
    {
      TimeFrameDefinitions this_time_frame_defs(frame_defs, 1);
      this_frame_exam_info.set_time_frame_definitions(this_time_frame_defs);
    }

    {
        char rest[50];
        sprintf(rest, "_f%dg1d0b0", current_frame_num);
        const string output_filename = output_filename_prefix + rest;

        proj_data_sptr =
                construct_proj_data(output, output_filename, this_frame_exam_info, template_proj_data_info_ptr);
    }

    for (int current_timing_pos_index = proj_data_sptr->get_min_tof_pos_num();
         current_timing_pos_index <= proj_data_sptr->get_max_tof_pos_num();
         current_timing_pos_index += 1)
    {
        for (int start_segment_index = proj_data_sptr->get_min_segment_num();
             start_segment_index <= proj_data_sptr->get_max_segment_num();
             start_segment_index += 1)
        {

            const int end_segment_index =
                    min( proj_data_sptr->get_max_segment_num()+1, start_segment_index + num_segments_in_memory) - 1;

            if (!interactive)
                allocate_segments(segments,
				  proj_data_sptr->get_min_tof_pos_num(),
                                  proj_data_sptr->get_max_tof_pos_num(),
                                  start_segment_index, end_segment_index,
				  proj_data_sptr->get_proj_data_info_sptr());

            for (int ax_num = proj_data_sptr->get_proj_data_info_sptr()->get_min_axial_pos_num(start_segment_index);
                 ax_num <= proj_data_sptr->get_proj_data_info_sptr()->get_max_axial_pos_num(start_segment_index);
                 ++ax_num)
            {
                for (int view_num = proj_data_sptr->get_proj_data_info_sptr()->get_min_view_num();
                     view_num <= proj_data_sptr->get_proj_data_info_sptr()->get_max_view_num(); ++view_num)
                {
                    for (int tang_num =  proj_data_sptr->get_proj_data_info_sptr()->get_min_tangential_pos_num();
                         tang_num <=  proj_data_sptr->get_proj_data_info_sptr()->get_max_tangential_pos_num();
                         ++tang_num)
                    {
                      (*(segments[current_timing_pos_index][start_segment_index]))[view_num][ax_num][tang_num] = current_timing_pos_index;
                    }
                }
            }

            if (!interactive)
                save_and_delete_segments(output, segments,
                                         proj_data_sptr->get_min_tof_pos_num(),
                                         proj_data_sptr->get_max_tof_pos_num(),
                                         start_segment_index, end_segment_index,
                                         *proj_data_sptr);
        } // end of for loop for segment range

    } // end of for loop for timing positions
#endif
}



/************************* Local helper routines *************************/


void 
allocate_segments(VectorWithOffset<VectorWithOffset<segment_type *> > & segments,
		  const int start_timing_pos_index, 
		  const int end_timing_pos_index,
		  const int start_segment_index, 
		  const int end_segment_index,
		  const shared_ptr<const ProjDataInfo> proj_data_info_sptr)
{

  for (int timing_pos_num=start_timing_pos_index ; timing_pos_num<=end_timing_pos_index; timing_pos_num++)
      for (int seg=start_segment_index ; seg<=end_segment_index; seg++)
  {
#ifdef USE_SegmentByView
    segments[timing_pos_num][seg] = new SegmentByView<elem_type>(
        proj_data_info_sptr->get_empty_segment_by_view (seg, false, timing_pos_num));
#else
    segments[timing_pos_num][seg] =
      new Array<3,elem_type>(IndexRange3D(0, proj_data_info_sptr->get_num_views()-1,
                      0, proj_data_info_sptr->get_num_axial_poss(seg)-1,
                      -(proj_data_info_sptr->get_num_tangential_poss()/2),
                      proj_data_info_sptr->get_num_tangential_poss()-(proj_data_info_sptr->get_num_tangential_poss()/2)-1));
#endif
  }
}

void 
save_and_delete_segments(shared_ptr<iostream>& output,
			 VectorWithOffset<VectorWithOffset<segment_type *> >& segments,
			 const int start_timing_pos_index, 
			 const int end_timing_pos_index,
			 const int start_segment_index, 
			 const int end_segment_index,
			 ProjData& proj_data)
{
  for (int timing_pos_num=start_timing_pos_index ; timing_pos_num<=end_timing_pos_index; timing_pos_num++)
    for (int seg=start_segment_index; seg<=end_segment_index; seg++)
      {
#ifdef USE_SegmentByView
	proj_data.set_segment(*segments[timing_pos_num][seg]);
#else
	(*segments[timing_pos_num][seg]).write_data(*output);
#endif
      delete segments[timing_pos_num][seg];    
      }
}



static
shared_ptr<ProjData>
construct_proj_data(shared_ptr<iostream>& output,
                    const string& output_filename, 
		    const ExamInfo& exam_info,
                    const shared_ptr<const ProjDataInfo>& proj_data_info_ptr)
{
  shared_ptr<ExamInfo> exam_info_sptr(new ExamInfo(exam_info));
#ifdef USE_SegmentByView
  shared_ptr<ProjData> proj_data_sptr;
  // don't need output stream in this case
  if (!proj_data_info_ptr->is_tof_data())
	  proj_data_sptr.reset(new ProjDataInterfile(exam_info_sptr,
                                                 proj_data_info_ptr, output_filename, ios::out,
                                                 ProjDataFromStream::Segment_View_AxialPos_TangPos,
                                                 OUTPUTNumericType));
  else
      proj_data_sptr.reset(new ProjDataInterfile(exam_info_sptr,
                                                 proj_data_info_ptr, output_filename, ios::out,
                                                 ProjDataFromStream::Timing_Segment_View_AxialPos_TangPos,
                                                 OUTPUTNumericType));

  return proj_data_sptr;
#else
  // this code would work for USE_SegmentByView as well, but the above is far simpler...
  vector<int> segment_sequence_in_stream(proj_data_info_ptr->get_num_segments());
  { 
    std::vector<int>::iterator current_segment_iter =
      segment_sequence_in_stream.begin();
    for (int segment_num=proj_data_info_ptr->get_min_segment_num();
         segment_num<=proj_data_info_ptr->get_max_segment_num();
         ++segment_num)
      *current_segment_iter++ = segment_num;
  }
  output = new fstream (output_filename.c_str(), ios::out|ios::binary);
  if (!*output)
    error("Error opening output file %s\n",output_filename.c_str());
  shared_ptr<ProjDataFromStream> proj_data_ptr(
					       new ProjDataFromStream(exam_info_sptr, proj_data_info_ptr, output, 
								      /*offset=*/std::streamoff(0), 
								      segment_sequence_in_stream,
								      ProjDataFromStream::Segment_View_AxialPos_TangPos,
								      OUTPUTNumericType));
  write_basic_interfile_PDFS_header(output_filename, *proj_data_ptr);
  return proj_data_ptr;  
#endif
}


END_NAMESPACE_STIR
