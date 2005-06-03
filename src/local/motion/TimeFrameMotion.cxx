//
// $Id$
//
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd
    Internal GE use only
*/
/*!
  \file
  \ingroup motion
  \brief stir::TimeFrameMotion

  \author Kris Thielemans
  $Date$
  $Revision$
*/

#include "local/stir/motion/TimeFrameMotion.h"
#include "stir/is_null_ptr.h"

START_NAMESPACE_STIR

void 
TimeFrameMotion::set_defaults()
{
  ro3d_ptr = 0;
  _reference_abs_time_sptr = 0;
  frame_num_to_process = -1;
  do_move_to_reference = true;
  scan_start_time_secs_since_1970_UTC=-1;
}

void 
TimeFrameMotion::initialise_keymap()
{

  parser.add_key("scan_start_time_secs_since_1970_UTC", 
		 &scan_start_time_secs_since_1970_UTC);
  parser.add_key("time frame definition filename",&frame_definition_filename);
  parser.add_parsing_key("time interval for reference position type", &_reference_abs_time_sptr);
  parser.add_key("move_to_reference", &do_move_to_reference);
  parser.add_key("frame_num_to_process", &frame_num_to_process);
  parser.add_parsing_key("Rigid Object 3D Motion Type", &ro3d_ptr); 
  parser.add_stop_key("END");
}

/*
TimeFrameMotion::
TimeFrameMotion(const char * const par_filename)
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
*/
bool
TimeFrameMotion::
post_processing()
{

  if (scan_start_time_secs_since_1970_UTC==-1)
    {
      warning("scan_start_time_secs_since_1970_UTC not set.\n"
	      "Will use relative time (to RigidObjectMotion object, which for Polaris means relative to the list mode data).");
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
	  ro3d_ptr->secs_since_1970_to_rel_time(sec_time);
      }
    }
  
  // handle time frame definitions etc

  if (frame_definition_filename.size()==0)
    {
      warning("Have to specify 'time frame_definition_filename'\n");
      return true;
    }

  frame_defs = TimeFrameDefinitions(frame_definition_filename);

  if (is_null_ptr(ro3d_ptr))
  {
    warning("Invalid Rigid Object 3D Motion object\n");
    return true;
  }

  if (frame_num_to_process!=-1 &&
      (frame_num_to_process<1 || 
       static_cast<unsigned>(frame_num_to_process)>frame_defs.get_num_frames()))
    {
      warning("Frame number should be between 1 and %d\n",
	      frame_defs.get_num_frames());
      return true;
    }

  // set transformation_to_reference_position
  if (is_null_ptr(_reference_abs_time_sptr))
    {
      warning("time interval for reference position is not set");
      return true;
    }
    {
      RigidObject3DTransformation av_motion = 
	ro3d_ptr->compute_average_motion(*_reference_abs_time_sptr);
      _transformation_to_reference_position =av_motion.inverse();    
    }

    set_frame_num_to_process(frame_num_to_process);

    return false;

}

void 
TimeFrameMotion::
move_to_reference(const bool value)
{
  do_move_to_reference=value;
}

int
TimeFrameMotion::
get_frame_num_to_process() const
{
  return frame_num_to_process;
}
void
TimeFrameMotion::
set_frame_num_to_process(const int value)
{
  frame_num_to_process=value;
  if (frame_num_to_process==-1)
    return;

  const double start_time = 
    frame_defs.get_start_time(frame_num_to_process) + scan_start_time;
  const double end_time = 
    frame_defs.get_end_time(frame_num_to_process) +  scan_start_time;
  cerr << "\nDoing frame " << frame_num_to_process
       << ": from " << start_time << " to " << end_time << endl;
  
  
  _current_rigid_object_transformation =
    ro3d_ptr->compute_average_motion_rel_time(start_time, end_time);
  
  _current_rigid_object_transformation = 
    compose(ro3d_ptr->get_transformation_to_scanner_coords(),
	    compose(_transformation_to_reference_position,
		    compose(_current_rigid_object_transformation,
			    ro3d_ptr->get_transformation_from_scanner_coords())));
  if (!do_move_to_reference)
    _current_rigid_object_transformation = 
      _current_rigid_object_transformation.inverse();
}

const RigidObject3DTransformation& 
TimeFrameMotion::
get_current_rigid_object_transformation() const
{
  return _current_rigid_object_transformation;
}

const TimeFrameDefinitions&
TimeFrameMotion::
get_time_frame_defs() const
{
  return frame_defs;
}

END_NAMESPACE_STIR
