//
// $Id$
//
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd
    This file is for internal GE use only
*/
/*!
  \file
  \ingroup motion_utilities
  \brief Utility to compute norm factors for motion corrected projection data

  See class documentation for stir::FindMCNormFactors for more info.

  \par Usage
\verbatim
  find_motion_corrected_norm_factors parameter_file
\endverbatim
  \author Kris Thielemans
  $Date$
  $Revision$
*/
#include "stir/ProjDataInterfile.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/CartesianCoordinate3D.h"

#include "local/stir/listmode/LmToProjData.h"
#include "local/stir/motion/RigidObject3DMotion.h"
#include "stir/TimeFrameDefinitions.h"
#include "stir/recon_buildblock/TrivialBinNormalisation.h"
#include "stir/Succeeded.h"
#include "stir/IO/write_data.h"
#include "stir/is_null_ptr.h"
#include "stir/round.h"

// TODO currently necessary, but needs to be replaced by ProjDataInMemory
#define USE_SegmentByView

#ifdef USE_SegmentByView
#include "stir/SegmentByView.h"
#else
#include "stir/Array.h"
#include "stir/IndexRange3D.h"
#endif

// set elem_type to what you want to use for the sinogram elements

#if defined(USE_SegmentByView) 
   typedef float elem_type;
#  define OUTPUTNumericType NumericType::FLOAT
#else
   typedef short elem_type;
#  define OUTPUTNumericType NumericType::SHORT
#endif

START_NAMESPACE_STIR

#ifdef USE_SegmentByView
typedef SegmentByView<elem_type> segment_type;
#endif
/******************** Prototypes  for local routines ************************/
// used for allocating segments.
// TODO replace by ProjDataInMemory

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

/*! \ingroup motion
  \brief Class to compute 'time-efficiency' factors for motino corrected projection data

  When list mode data is binned into 3d sinograms using motion correction, (or
  when a 3d sinograms is motion corrected), the resulting sinogram is not 
  'consistent' due to some LORs not being measured during the whole frame.
  This is discussed in some detail in<br>
  K. Thielemans, S. Mustafovic, L. Schnorr, 
  <i>Image Reconstruction of Motion Corrected Sinograms</i>, 
  poster at IEEE Medical Imaging Conf. 2003,
  available at http://www.hammersmithimanet.com/~kris/papers/.

  This class computes these 'time-efficiency' factors.

  See general MC doc for how the LMC method works.

  \par Format of parameter file
  \verbatim
FindMCNormFactors Parameters :=
; output name
; filenames will be constructed by appending _f#g1d0b0.hs and .s
; where # is the frame number
output filename prefix:= output

; file to get frame definitions (see doc for TimeFrameDefinitions)
time frame_definition file:=some_ECAT7.S
; file that will be used to get dimensions/scanner etc
; can usually be the same ECAT7 file as above
template_projdata:= projdata
; frame to do in this run (if -1 all frames will be done)
time frame number := 1

; next allows you to do only a few segments (defaults to all in template)
;maximum absolute segment number to process:=0

; object specifying motion data
; example given for Polaris
; warning: the Polaris parameters might change
; (see doc for RigidObject3DMotionFromPolaris for up-to-date info)
Rigid Object 3D Motion Type:=Motion From Polaris
Rigid Object 3D Motion From Polaris Parameters:=
mt filename:=H02745.mt
list_mode_filename:= H02745_lm1
attenuation_filename:=H02745_tr.a
transmission_duration:=300
transformation_from_scanner_coordinates_filename:=966/transformation_from_scanner_to_polaris
End Rigid Object 3D Motion From Polaris:=

; experimental support for method where the usual detection efficiencies
; are taken into account here, and not during the list mode binning
; default is not to use this
do pre normalisation := 0
Bin Normalisation type := supported_normalisation_type


; specify number of intervals that will be taken in this frame

; default duration in secs
default time interval:=5
minimum number of time intervals per frame:= 1
maximum number of time intervals per frame:=1

END:=
  \endverbatim
*/
class FindMCNormFactors : public ParsingObject
{
public:
  FindMCNormFactors(const char * const par_filename);

  TimeFrameDefinitions frame_defs;

  virtual void process_data();
  
protected:

  
  //! parsing functions
  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();

  //! parsing variables
  string output_filename_prefix;
  string template_proj_data_name;
  string frame_definition_filename;
  int max_segment_num_to_process;

  shared_ptr<ProjDataInfo> template_proj_data_info_ptr;
  shared_ptr<ProjDataInfo> proj_data_info_uncompressed_ptr;
  const ProjDataInfoCylindricalNoArcCorr * proj_data_info_cyl_uncompressed_ptr;
  shared_ptr<Scanner> scanner_ptr;
  bool do_pre_normalisation;
  

  bool do_time_frame;

     
private:
  int frame_num;
  shared_ptr<RigidObject3DMotion> ro3d_ptr;
  shared_ptr<BinNormalisation> normalisation_ptr;
 
  double time_interval;
  int min_num_time_intervals_per_frame;
  int max_num_time_intervals_per_frame;
};

void 
FindMCNormFactors::set_defaults()
{
  max_segment_num_to_process = -1;
  ro3d_ptr = 0;
  normalisation_ptr = 0;
  do_pre_normalisation = true;
  time_interval=1; 
  min_num_time_intervals_per_frame = 1;
  max_num_time_intervals_per_frame = 100;
  frame_num = -1;
}

void 
FindMCNormFactors::initialise_keymap()
{

  parser.add_start_key("FindMCNormFactors Parameters");

  parser.add_key("template_projdata", &template_proj_data_name);
  parser.add_key("maximum absolute segment number to process", &max_segment_num_to_process); 
  parser.add_key("time frame_definition file",&frame_definition_filename);
  parser.add_key("time frame number", &frame_num);

  parser.add_key("output filename prefix",&output_filename_prefix);
  parser.add_parsing_key("Rigid Object 3D Motion Type", &ro3d_ptr); 
  parser.add_parsing_key("Bin Normalisation type", &normalisation_ptr);
  parser.add_key("do pre normalisation", &do_pre_normalisation);
  parser.add_key("default time interval", &time_interval);
  parser.add_key("minimum number of time intervals per frame", &min_num_time_intervals_per_frame);
  parser.add_key("maximum number of time intervals per frame", &max_num_time_intervals_per_frame);
  parser.add_stop_key("END");
}

FindMCNormFactors::
FindMCNormFactors(const char * const par_filename)
{
  set_defaults();
  if (par_filename!=0)
    {
      if (parse(par_filename) == false)
	error("Please correct parameter file");
    }
  else
    ask_parameters();

}

bool
FindMCNormFactors::
post_processing()
{
   

  if (output_filename_prefix.size()==0)
    {
      warning("You have to specify an output_filename_prefix\n");
      return true;
    }


  if (template_proj_data_name.size()==0)
    {
      warning("You have to specify template_projdata\n");
      return true;
    }
  shared_ptr<ProjData> template_proj_data_ptr =
    ProjData::read_from_file(template_proj_data_name);

  template_proj_data_info_ptr = 
    template_proj_data_ptr->get_proj_data_info_ptr()->clone();

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


    scanner_ptr = 
    new Scanner(*template_proj_data_info_ptr->get_scanner_ptr());

  // TODO this won't work for the HiDAC or so
  proj_data_info_uncompressed_ptr =
    ProjDataInfo::ProjDataInfoCTI(scanner_ptr, 
                  1, scanner_ptr->get_num_rings()-1,
                  scanner_ptr->get_num_detectors_per_ring()/2,
                  scanner_ptr->get_default_num_arccorrected_bins(), 
                  false);
  proj_data_info_cyl_uncompressed_ptr =
    dynamic_cast<ProjDataInfoCylindricalNoArcCorr const *>
    (proj_data_info_uncompressed_ptr.get());


  if (!do_pre_normalisation && is_null_ptr(normalisation_ptr))
    {
      //normalisation_ptr = new TrivialBinNormalisation;
      warning("Invalid normalisation object\n");
      return true;
    }
  if (!do_pre_normalisation &&
      normalisation_ptr->set_up(proj_data_info_uncompressed_ptr)
       != Succeeded::yes)
    {
      warning("set-up of normalisation failed\n");
      return true;
    }

  // handle time frame definitions etc

  do_time_frame = true;

  if (do_time_frame && frame_definition_filename.size()==0)
    {
      warning("Have to specify either 'time frame_definition_filename' or 'num_events_to_store'\n");
      return true;
    }

  if (frame_definition_filename.size()!=0)
    frame_defs = TimeFrameDefinitions(frame_definition_filename);
  else
    {
      // make a single frame starting from 0. End value will be ignored.
      vector<pair<double, double> > frame_times(1, pair<double,double>(0,1));
      frame_defs = TimeFrameDefinitions(frame_times);
    }
  if (frame_num != -1 && 
      (frame_num<1 || unsigned(frame_num)> frame_defs.get_num_frames()))
    {
      warning("'time frame num (%u) should be either -1 or between 1 and the number of frames (%u)",
	      frame_num, frame_defs.get_num_frames());
      return true;
    }
	
  if (is_null_ptr(ro3d_ptr))
  {
    warning("Invalid Rigid Object 3D Motion object\n");
    return true;
  }


#if 0
  if (!ro3d_ptr->is_synchronised())
    {
      warning("You have to specify an input_filename (or an explicit time offset) for the motion object\n");
	  return true;
    }
#endif

  return false;
}

 

void 
FindMCNormFactors::process_data()
{
#ifdef NEW_ROT
  cerr << "Using NEW_ROT\n";
#else
  cerr << "Using original ROT\n";
#endif

  VectorWithOffset<segment_type *> 
    segments (template_proj_data_info_ptr->get_min_segment_num(), 
	      template_proj_data_info_ptr->get_max_segment_num());

  const unsigned min_frame_num =
    frame_num==-1 ? 1 : unsigned(frame_num);
  const unsigned max_frame_num =
    frame_num==-1 ? frame_defs.get_num_frames() : unsigned(frame_num);

  for (unsigned int current_frame_num = min_frame_num;
       current_frame_num<= max_frame_num;
       ++current_frame_num)
    {
      const double start_time = frame_defs.get_start_time(current_frame_num);
      const double end_time = frame_defs.get_end_time(current_frame_num);
      const double frame_duration = end_time - start_time;
      const int num_time_intervals_this_frame =
	max(min(round(frame_duration/time_interval),
		max_num_time_intervals_per_frame),
	    min_num_time_intervals_per_frame);
      const double time_interval_this_frame =
	frame_duration / num_time_intervals_this_frame;

      cerr << "\nDoing frame " << current_frame_num
	   << ": from " << start_time << " to " << end_time 
	   << " with " << num_time_intervals_this_frame
	   << " time intervals of length "
	   << time_interval_this_frame
	   << endl;


      //*********** open output file
	shared_ptr<iostream> output;
	shared_ptr<ProjData> out_proj_data_ptr;

	{
	  char rest[50];
	  sprintf(rest, "_f%dg1d0b0", current_frame_num);
	  const string output_filename = output_filename_prefix + rest;
      
	  out_proj_data_ptr = 
	    construct_proj_data(output, output_filename, template_proj_data_info_ptr);
	}

	allocate_segments(segments, 
			  template_proj_data_info_ptr->get_min_segment_num(), 
			  template_proj_data_info_ptr->get_max_segment_num(),
			  template_proj_data_info_ptr.get());

	const int start_segment_index = template_proj_data_info_ptr->get_min_segment_num();
	const int end_segment_index = template_proj_data_info_ptr->get_max_segment_num();

	 
	const ProjDataInfoCylindricalNoArcCorr * const out_proj_data_info_ptr =
	  dynamic_cast<ProjDataInfoCylindricalNoArcCorr const * >
	  (out_proj_data_ptr->get_proj_data_info_ptr());
	if (out_proj_data_info_ptr== NULL)
	  {
	    error("works only on  proj_data_info of "
		  "type ProjDataInfoCylindricalNoArcCorr\n");
	  }

	int current_num_time_intervals=0; 
	cerr << "Doing time intervals: ";
	for (double current_time = start_time;
	     current_time<=end_time; 
	     current_time+=time_interval_this_frame)
	  {
	    if (++current_num_time_intervals > num_time_intervals_this_frame)
	      break;
	    cerr << '(' << current_time << '-' << current_time+time_interval_this_frame << ") ";

	    if (current_time+time_interval_this_frame > end_time + time_interval_this_frame*.01)
	      error("\ntime interval goes beyond end of frame. Check code!\n");
	    RigidObject3DTransformation ro3dtrans =
	      ro3d_ptr->compute_average_motion_rel_time(current_time, current_time+time_interval_this_frame);
	    ro3dtrans = 
	      compose(ro3d_ptr->get_transformation_to_scanner_coords(),
		      compose(ro3d_ptr->get_transformation_to_reference_position(),
			      compose(ro3dtrans,
				ro3d_ptr->get_transformation_from_scanner_coords())));
            
	    for (int in_segment_num = proj_data_info_uncompressed_ptr->get_min_segment_num(); 
		 in_segment_num <= proj_data_info_uncompressed_ptr->get_max_segment_num();
		 ++in_segment_num)
	      {


		for (int in_ax_pos_num = proj_data_info_uncompressed_ptr->get_min_axial_pos_num(in_segment_num); 
		     in_ax_pos_num  <= proj_data_info_uncompressed_ptr->get_max_axial_pos_num(in_segment_num);
		     ++in_ax_pos_num )
		  {
	      
		    for (int in_view_num=proj_data_info_uncompressed_ptr->get_min_view_num();
			 in_view_num <= proj_data_info_uncompressed_ptr->get_max_view_num();
			 ++in_view_num)
		      {
		  
			for (int in_tangential_pos_num=proj_data_info_uncompressed_ptr->get_min_tangential_pos_num();
			     in_tangential_pos_num <= proj_data_info_uncompressed_ptr->get_max_tangential_pos_num();
			     ++in_tangential_pos_num)
			  {
			    const Bin original_bin(in_segment_num,in_view_num,in_ax_pos_num, in_tangential_pos_num, 1);
			    // find new bin position
			    Bin bin = original_bin;
			 
			    ro3dtrans.transform_bin(bin, 
						      *out_proj_data_info_ptr,
						      *proj_data_info_cyl_uncompressed_ptr);
#if 0
			      if ((bin.axial_pos_num()-original_bin.axial_pos_num())> 0.0001 ||  
				  (bin.segment_num()-original_bin.segment_num())> 0.0001 ||
				   (bin.tangential_pos_num()-original_bin.tangential_pos_num()) > 0.0001||
				   (bin.view_num() -original_bin.view_num())> 0.0001) 

  {
    Quaternion<float> quat = ro3dtrans.get_quaternion();
    cerr << quat[1] << "   " << quat[2]<<  "   " << quat[3]<< "   " << quat[4]<< endl;
    CartesianCoordinate3D<float> trans=ro3dtrans.get_translation();
    cerr <<  trans.z() << "    " <<  trans.y() << "   " << trans.x() << endl;
    cerr << " Start" << endl;
cerr << " Original bin is " << original_bin.segment_num() << "   " << original_bin.axial_pos_num() << "   " << original_bin.view_num() << "    "  << original_bin.tangential_pos_num() << endl;
cerr << " Transformed  bin is " << bin.segment_num() << "   " << bin.axial_pos_num() << "   " << bin.view_num() << "    "  << bin.tangential_pos_num() << endl;

 cerr << " End" << endl;
  }
#endif
			    if (bin.get_bin_value()>0
				&& bin.tangential_pos_num()>= out_proj_data_ptr->get_min_tangential_pos_num()
				&& bin.tangential_pos_num()<= out_proj_data_ptr->get_max_tangential_pos_num()
				&& bin.axial_pos_num()>=out_proj_data_ptr->get_min_axial_pos_num(bin.segment_num())
				&& bin.axial_pos_num()<=out_proj_data_ptr->get_max_axial_pos_num(bin.segment_num())
				) 
			      {
				assert(bin.view_num()>=out_proj_data_ptr->get_min_view_num());
				assert(bin.view_num()<=out_proj_data_ptr->get_max_view_num());
				
				// now check if we have its segment in memory
				if (bin.segment_num() >= start_segment_index && bin.segment_num()<=end_segment_index)
				  {
				    // TODO remove scale factor
				    // it's there to compensate what we have in LmToProjDataWithMC
				    if (do_pre_normalisation)
				      {
				    (*segments[bin.segment_num()])[bin.view_num()][bin.axial_pos_num()][bin.tangential_pos_num()] += 
				      1.F/
				      (out_proj_data_info_ptr->
				       get_num_ring_pairs_for_segment_axial_pos_num(bin.segment_num(),
										    bin.axial_pos_num())*
				       out_proj_data_info_ptr->get_view_mashing_factor());
				      }
				    else
				      {
				      (*segments[bin.segment_num()])[bin.view_num()][bin.axial_pos_num()][bin.tangential_pos_num()] += 
				      normalisation_ptr->
					get_bin_efficiency(original_bin,start_time,end_time);

				      }
				    
				  }
			      }
			  }
		      }
		  }
	      }
	  }
	// decrease our counter of the number of time intervals to set it to
	// the number we actually had
	--current_num_time_intervals;
	if (current_num_time_intervals != num_time_intervals_this_frame)
	  warning("\nUnexpected number of time intervals %d, should be %d",
		  current_num_time_intervals, num_time_intervals_this_frame);
	for (int segment_num=start_segment_index; segment_num<=end_segment_index; ++segment_num)
	  {
	    if (current_num_time_intervals>0)
	      (*(segments[segment_num])) /= current_num_time_intervals;
	    // add constant to avoid division by 0 later.
	    (*(segments[segment_num])) +=.00001;
	  }
	save_and_delete_segments(output, segments, 
				 start_segment_index, end_segment_index, 
				 *out_proj_data_ptr);  
		   
    }
  
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
      write_data(*output, (*segments[seg]));
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



USING_NAMESPACE_STIR

int main(int argc, char * argv[])
{
  
  if (argc!=1 && argc!=2) {
    cerr << "Usage: " << argv[0] << " [par_file]\n";
    exit(EXIT_FAILURE);
  }
  FindMCNormFactors application(argc==2 ? argv[1] : 0);
  application.process_data();

  return EXIT_SUCCESS;
}
