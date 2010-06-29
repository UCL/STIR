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
  _ro3d_sptr = 0;
  _reference_abs_time_sptr = 0;
  _frame_num_to_process = -1;
  _do_move_to_reference = true;
  _scan_start_time_secs_since_1970_UTC=-1;
  _frame_defs=TimeFrameDefinitions();
}

void 
TimeFrameMotion::initialise_keymap()
{

  parser.add_key("scan_start_time_secs_since_1970_UTC", 
		 &_scan_start_time_secs_since_1970_UTC);
  parser.add_key("time frame definition filename",&_frame_definition_filename);
  parser.add_parsing_key("time interval for reference position type", &_reference_abs_time_sptr);
  parser.add_key("move_to_reference", &_do_move_to_reference);
  parser.add_key("frame_num_to_process", &_frame_num_to_process);
  parser.add_parsing_key("Rigid Object 3D Motion Type", &_ro3d_sptr); 
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

  if (_scan_start_time_secs_since_1970_UTC==-1)
    {
      warning("scan_start_time_secs_since_1970_UTC not set.\n"
	      "Frame definitions will be assumed to use the same reference point as the tracker (which for Polaris means the start of the list mode data).");
      _scan_start_time = 0;
    }
  else 
    {
      if (_scan_start_time_secs_since_1970_UTC<1000)
	{
	  warning("scan_start_time_secs_since_1970_UTC too small");
	  return true;
	}
      {
	// save _scan_start_time as rel_time as reported by ro3d
	time_t sec_time = _scan_start_time_secs_since_1970_UTC;
	
	_scan_start_time = 
	  _ro3d_sptr->secs_since_1970_to_rel_time(sec_time);
      }
    }
  
  // handle time frame definitions etc

  if (_frame_definition_filename.size()==0)
    {
      warning("Have to specify 'time frame_definition_filename'\n");
      return true;
    }

  _frame_defs = TimeFrameDefinitions(_frame_definition_filename);

  if (is_null_ptr(_ro3d_sptr))
  {
    warning("Invalid Rigid Object 3D Motion object\n");
    return true;
  }

  if (_frame_num_to_process!=-1 &&
      (_frame_num_to_process<1 || 
       static_cast<unsigned>(_frame_num_to_process)>_frame_defs.get_num_frames()))
    {
      warning("Frame number should be between 1 and %d\n",
	      _frame_defs.get_num_frames());
      return true;
    }

  // set transformation_to_reference_position
  if (is_null_ptr(_reference_abs_time_sptr))
    {
      warning("time interval for reference position is not set");
      return true;
    }
    {
      const RigidObject3DTransformation av_motion = 
	_ro3d_sptr->compute_average_motion_in_scanner_coords(*_reference_abs_time_sptr);
      _transformation_to_reference_position =av_motion.inverse();    
    }

    set_frame_num_to_process(_frame_num_to_process);

    return false;

}

void 
TimeFrameMotion::
move_to_reference(const bool value)
{
  _do_move_to_reference=value;
}

int
TimeFrameMotion::
get_frame_num_to_process() const
{
  return _frame_num_to_process;
}
void
TimeFrameMotion::
set_frame_num_to_process(const int value)
{
  _frame_num_to_process=value;
  if (_frame_num_to_process==-1)
    return;

  const double start_time = 
    this->get_frame_start_time(_frame_num_to_process);
  const double end_time = 
    this->get_frame_end_time(_frame_num_to_process);
  
  _current_rigid_object_transformation =
    compose(_transformation_to_reference_position,
	    _ro3d_sptr->
	    compute_average_motion_in_scanner_coords_rel_time(start_time, end_time));
  if (!_do_move_to_reference)
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
  return _frame_defs;
}

END_NAMESPACE_STIR
