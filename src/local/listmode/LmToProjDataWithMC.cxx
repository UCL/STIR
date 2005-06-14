/*!
  \file
  \ingroup listmode
  \brief Implementation of class LmToProjDataWithMC
  \author Sanida Mustafovic
  \author Kris Thielemans
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/
#include "local/stir/listmode/LmToProjDataWithMC.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/listmode/CListRecordECAT966.h"
#include "stir/IO/stir_ecat7.h"
#include "stir/Succeeded.h"
#include <time.h>
#include "stir/is_null_ptr.h"
#include "stir/stream.h"

//#define FRAME_BASED_DT_CORR

START_NAMESPACE_STIR

void 
LmToProjDataWithMC::set_defaults()
{
  LmToProjData::set_defaults();
  reference_position_is_average_position_in_frame = false;
  _reference_abs_time_sptr = 0;
  ro3d_ptr = 0; 
}

void 
LmToProjDataWithMC::initialise_keymap()
{
  LmToProjData::initialise_keymap();
  parser.add_start_key("LmToProjDataWithMC Parameters");
  parser.add_key("reference_position_is_average_position_in_frame", 
		 &reference_position_is_average_position_in_frame);
  parser.add_parsing_key("time interval for reference position type", 
			 &_reference_abs_time_sptr);
  parser.add_parsing_key("Rigid Object 3D Motion Type", &ro3d_ptr); 
}

LmToProjDataWithMC::
LmToProjDataWithMC(const char * const par_filename)
{
  set_defaults();
  if (par_filename!=0)
    parse(par_filename) ;
  else
    ask_parameters();

}

bool
LmToProjDataWithMC::
post_processing()
{
  if (LmToProjData::post_processing())
    return true;
   
  if (is_null_ptr(ro3d_ptr))
  {
    warning("Invalid Rigid Object 3D Motion object\n");
    return true;
  }

  if (reference_position_is_average_position_in_frame)
    {
      if (!is_null_ptr(_reference_abs_time_sptr))
	{
	  warning("time interval for reference position is set, but you asked for average over each frame");
	  return true;
	}
    }
  else
    {
      // set transformation_to_reference_position
      if (is_null_ptr(_reference_abs_time_sptr))
	{
	  warning("time interval for reference position is not set");
	  return true;
	}
      {
	const RigidObject3DTransformation av_motion = 
	  this->ro3d_ptr->
	  compute_average_motion_in_scanner_coords(*_reference_abs_time_sptr);
	cerr << "Reference quaternion:  " << av_motion.get_quaternion()<<endl;
	cerr << "Reference translation:  " << av_motion.get_translation()<<endl;
	_transformation_to_reference_position =av_motion.inverse();    
      }
    }

#ifdef FRAME_BASED_DT_CORR
  cerr << "LmToProjDataWithMC Using FRAME_BASED_DT_CORR\n";
#else
  cerr << "LmToProjDataWithMC NOT Using FRAME_BASED_DT_CORR\n";
#endif
#ifdef NEW_ROT
  cerr << "and NEW_ROT\n";
#else
  cerr << "and original ROT\n";
#endif
  return false;
}

void
LmToProjDataWithMC::
start_new_time_frame(const unsigned int current_frame_num)
{
  LmToProjData::start_new_time_frame(current_frame_num);
  if (reference_position_is_average_position_in_frame)
    {
     const double start_time = frame_defs.get_start_time(current_frame_num);
     const double end_time = frame_defs.get_end_time(current_frame_num);
     const RigidObject3DTransformation av_motion = 
       this->ro3d_ptr->
       compute_average_motion_in_scanner_coords_rel_time(start_time, end_time);
     cerr << "Reference quaternion:  " << av_motion.get_quaternion()<<endl;
     cerr << "Reference translation:  " << av_motion.get_translation()<<endl;
     _transformation_to_reference_position =av_motion.inverse();    
    }
}

void
LmToProjDataWithMC::
process_new_time_event(const CListTime& time_event)
{
  assert(fabs(current_time - time_event.get_time_in_secs())<.0001);     
  this->ro3dtrans = 
      compose(this->_transformation_to_reference_position,
	      this->ro3d_ptr->get_motion_in_scanner_coords_rel_time(current_time));

}

void 
LmToProjDataWithMC::get_bin_from_event(Bin& bin, const CListEvent& event_of_general_type) const
{
  const CListRecordECAT966& record = 
    static_cast<CListRecordECAT966 const&>(event_of_general_type);// TODO get rid of this

  const ProjDataInfoCylindricalNoArcCorr& proj_data_info =


  static_cast<const ProjDataInfoCylindricalNoArcCorr&>(*template_proj_data_info_ptr);
#ifndef FRAME_BASED_DT_CORR
  const double start_time = current_time;
  const double end_time = current_time;
#else
  const double start_time = frame_defs.get_start_time(current_frame_num);
  const double end_time =frame_defs.get_end_time(current_frame_num);
#endif

  record.get_uncompressed_bin(bin);
  const float bin_efficiency = normalisation_ptr->get_bin_efficiency(bin,start_time,end_time);
   
  //Do the motion correction
  ro3dtrans.transform_bin(bin,
			  proj_data_info,
			  *record.get_uncompressed_proj_data_info_sptr());

  if (bin.get_bin_value() > 0)
    {
      if (do_pre_normalisation)
	{
	  // now normalise event taking into account the
	  // normalisation factor before motion correction
	  // Note: this normalisation is not really correct
	  // we need to take the number of uncompressed bins that
	  // contribute to this bin into account (will be done in 
	  // do_post_normalisation).
	  // In addition, there is time-based normalisation.
	  // See Thielemans et al, Proc. MIC 2003
	  bin.set_bin_value(1/bin_efficiency);
	}
      else
	{
	  bin.set_bin_value(1);
	}
    }
  
}


END_NAMESPACE_STIR
