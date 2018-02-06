//
//
/*
    Copyright (C) 2003- 2012 , Hammersmith Imanet Ltd
    For GE Internal use only
*/
/*!
  \file
  \ingroup motion
  \brief Implementation of class stir::MatchTrackerAndScanner.
  \author Kris Thielemans

  
*/
#include "local/stir/motion/MatchTrackerAndScanner.h"
#include "stir/stream.h"
#include "stir/round.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/DiscretisedDensityOnCartesianGrid.h"
#include "stir/is_null_ptr.h"
#include "stir/index_at_maximum.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/IO/read_from_file.h"
#include "stir/centre_of_gravity.h"
#include "stir/thresholding.h"
#include <vector>
#include <algorithm>
#include <iostream>
#include <cmath>
# ifdef BOOST_NO_STDC_NAMESPACE
namespace std { using ::sqrt; }
# endif

START_NAMESPACE_STIR


void 
MatchTrackerAndScanner::set_defaults()
{
  this->_ro3d_sptr.reset();
  this->scan_start_time_secs_since_1970_UTC=-1;
  this->relative_threshold = .1F;
}

void 
MatchTrackerAndScanner::initialise_keymap()
{

  parser.add_start_key("Match Tracker and Scanner Parameters");
  parser.add_key("scan_start_time_secs_since_1970_UTC", 
		 &this->scan_start_time_secs_since_1970_UTC);
  parser.add_key("time frame definition filename",&this->frame_definition_filename);
  parser.add_parsing_key("Rigid Object 3D Motion Type", &this->_ro3d_sptr); 
  parser.add_key("image_filename_prefix", &this->_image_filename_prefix);
  parser.add_key("relative_threshold", &this->relative_threshold);

  parser.add_stop_key("END");
}

MatchTrackerAndScanner::
MatchTrackerAndScanner(const char * const par_filename)
{
  set_defaults();
  if (par_filename!=0)
    {
      if (parse(par_filename)==false)
	exit(EXIT_FAILURE);
    }
  else
    ask_parameters();

}

bool
MatchTrackerAndScanner::
post_processing()
{

  if (scan_start_time_secs_since_1970_UTC==-1)
    {
      warning("scan_start_time_secs_since_1970_UTC not set.\n"
	      "Will use relative time (to RigidObjectMotion object, which for Polaris means relative to the start of the scan data you use for synchronisation).");
      scan_start_time = 0;
    }
  else 
    {
      if (scan_start_time_secs_since_1970_UTC<1000)
	{
	  warning("scan_start_time_secs_since_1970_UTC too small");
	  return true;
	}
      {
	// convert to time_in_secs since midnight
	time_t sec_time = scan_start_time_secs_since_1970_UTC;
	
	scan_start_time = 
	  _ro3d_sptr->secs_since_1970_to_rel_time(sec_time);
      }
    }
  
  // handle time frame definitions etc

  if (frame_definition_filename.size()==0)
    {
      warning("Have to specify 'time frame_definition_filename'");
      return true;
    }

  frame_defs = TimeFrameDefinitions(frame_definition_filename);

  if (is_null_ptr(_ro3d_sptr))
  {
    warning("Invalid Rigid Object 3D Motion object");
    return true;
  }


  if (_image_filename_prefix.size()==0)
  {
    warning("have to specify 'image_filename_prefix'");
    return true;
  }

  if (this->relative_threshold<0.F || this->relative_threshold>1.F)
  {
    warning("this->relative_threshold has to be between 0 and 1");
    return true;
  }
  return false;

}

const TimeFrameDefinitions&
MatchTrackerAndScanner::
get_time_frame_defs() const
{
  return frame_defs;
}


Succeeded
MatchTrackerAndScanner::
run()
{
  std::vector<CartesianCoordinate3D<float> > polaris_points;
  std::vector<CartesianCoordinate3D<float> > positions_in_scanner;

  std::cout << "\nI will now read the images and tracker data for each time frame.\n"
	    << "I will report intra-frame movement as the stddev w.r.t. mean position\n"
	    << "of the marker according to the tracker data. This is a value in mm.\n";

  for (unsigned current_frame_num=1U;
       current_frame_num<=this->get_time_frame_defs().get_num_frames(); 
       ++current_frame_num)
    {
      // read image and find maximum
      CartesianCoordinate3D<float> location_of_image_max_in_mm;
      {
	char rest[50];
	sprintf(rest, "_f%ug1d0b0.hv", current_frame_num);
	const string input_filename = this->get_image_filename_prefix() + rest;
	
	shared_ptr< DiscretisedDensity<3,float> >  
	  input_image_sptr(read_from_file<DiscretisedDensity<3,float> >(input_filename));


#if 0
        // old code that used the location of the maximum
	const DiscretisedDensityOnCartesianGrid <3,float>*  input_image_cartesian_ptr = 
	  dynamic_cast< DiscretisedDensityOnCartesianGrid<3,float>*  > (input_image_sptr.get());

	if (input_image_cartesian_ptr== 0)
	  {
	    error("Image '%s' should be on a cartesian grid",
		  input_filename.c_str());
	  }
	const CartesianCoordinate3D<float> grid_spacing = 
	  input_image_cartesian_ptr->get_grid_spacing();

	const BasicCoordinate<3,int> max_index = 
	  indices_at_maximum(*input_image_sptr);

	location_of_image_max_in_mm =
	  grid_spacing * BasicCoordinate<3,float>(max_index);
#else
        // new code that uses centre of gravity
	const VoxelsOnCartesianGrid <float>*  input_image_cartesian_ptr = 
	  dynamic_cast< VoxelsOnCartesianGrid<float>*  > (input_image_sptr.get());

	if (input_image_cartesian_ptr== 0)
	  {
	    error("Image '%s' should be voxels on a cartesian grid",
		  input_filename.c_str());
	  }
        const float threshold = 
          input_image_sptr->find_max() * this->relative_threshold;
        threshold_lower(input_image_sptr->begin_all(), input_image_sptr->end_all(), 
                        threshold);
        (*input_image_sptr) -= threshold;
        location_of_image_max_in_mm = 
          find_centre_of_gravity_in_mm(*input_image_cartesian_ptr);
#endif
      }

      // now go through tracker data for this frame
      {
	const double start_time = 
	  this->get_frame_start_time(current_frame_num);
	const double end_time = 
	  this->get_frame_end_time(current_frame_num);

	cerr << "\nDoing frame " << current_frame_num
	     << ": from " << start_time << " to " << end_time << endl;

	const std::vector<double> sample_times =
	  this->get_motion().
	  get_rel_time_of_samples(start_time, end_time);

	if (sample_times.size() == 0)
	  error("No tracker samples between %g and %g (relative to scan start)",
		start_time, end_time);

	// some variables that will be used to compute the stddev over the frame
	// to check intra-frame movement
	CartesianCoordinate3D<float> sum_location_in_tracker_coords(0,0,0);
	CartesianCoordinate3D<float> sum_square_location_in_tracker_coords(0,0,0);
	CartesianCoordinate3D<float> first_location_in_tracker_coords =
	      this->get_motion().
	      get_motion_in_tracker_coords_rel_time(sample_times[0]).inverse().
	      transform_point(CartesianCoordinate3D<float>(0,0,0));

	for (std::vector<double>::const_iterator iter=sample_times.begin();
	     iter != sample_times.end();
	     ++iter)
	  {
	    CartesianCoordinate3D<float> location_in_tracker_coords =
	      this->get_motion().
	      get_motion_in_tracker_coords_rel_time(*iter).inverse().
	      transform_point(CartesianCoordinate3D<float>(0,0,0));
	    polaris_points.push_back(location_in_tracker_coords);
	    positions_in_scanner.push_back(location_of_image_max_in_mm);

	    sum_location_in_tracker_coords +=
	      location_in_tracker_coords - first_location_in_tracker_coords;
	    sum_square_location_in_tracker_coords +=
	      square(location_in_tracker_coords - first_location_in_tracker_coords);
	  }
	// check if frame is uniform
	const unsigned num_samples = sample_times.size();
	const CartesianCoordinate3D<float> variance =
	  (sum_square_location_in_tracker_coords -
	   square(sum_location_in_tracker_coords)/num_samples)/
	  (num_samples-1);
	//std::cerr << sum_location_in_tracker_coords/num_samples
	//	  << sum_square_location_in_tracker_coords/num_samples
	//	  << variance;
	// note: threshold with 0 before sqrt to avoid problems with rounding errors
	const double stddev = 
	  std::sqrt(std::max(0.F,(variance[1]+variance[2]+variance[3])/3));

	if (stddev>2)
	  warning("Intra-frame motion for frame %d is too large:  %g",
		  current_frame_num, stddev);
	else
	  std::cerr << "Intra-frame motion for frame " << current_frame_num
		    << " : " << stddev;	
      }
    } // end of loop over frames

  //std::cout << positions_in_scanner;
  //std::cout << polaris_points;

  // now find match
  RigidObject3DTransformation transformation;
  if (RigidObject3DTransformation::
      find_closest_transformation(this->_transformation_from_scanner_coords,
				  positions_in_scanner.begin(), positions_in_scanner.end(), 
				  polaris_points.begin(),
				  Quaternion<float>(1,0,0,0)) ==
      Succeeded::no)
    {
      warning("Could not find match"); // note: find_closest_transformation writes some more info
      return Succeeded::no;
    }
  
  const double RMSE =
    RigidObject3DTransformation::RMSE(this->_transformation_from_scanner_coords, 
				      positions_in_scanner.begin(), positions_in_scanner.end(), 
				      polaris_points.begin());

  std::cout << "\n\nResult for transformation from scanner to tracker:\n\t" 
	    << this->_transformation_from_scanner_coords;
  std::cout << "\nRMSE (in mm) = " << RMSE
	    << '\n';
  if (RMSE>4)
    warning("RMSE is rather large. I'm expecting a value around 1.5 mm");

  std::cout << "If this is ok, edit your file specifying the transformation.\n";
  return Succeeded::yes;
}

END_NAMESPACE_STIR
