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
  _reference_abs_time_sptr = 0;
  ro3d_ptr = 0; 
}

void 
LmToProjDataWithMC::initialise_keymap()
{
  LmToProjData::initialise_keymap();
  parser.add_start_key("LmToProjDataWithMC Parameters");
  parser.add_parsing_key("time interval for reference position type", &_reference_abs_time_sptr);
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
#if 0
  // TODO move to RigidObject3DMotion, bu tit does not necessarily have the list mode data
  if (!ro3d_ptr->is_synchronised())
    ro3d_ptr->synchronise(*lm_data_ptr);
#endif

  // set transformation_to_reference_position
  if (is_null_ptr(_reference_abs_time_sptr))
    {
      warning("time interval for reference position is not set");
      return true;
    }
    {
      RigidObject3DTransformation av_motion = 
	ro3d_ptr->compute_average_motion(*_reference_abs_time_sptr);
      cerr << "Reference quaternion:  " << av_motion.get_quaternion()<<endl;
      cerr << "Reference translation:  " << av_motion.get_translation()<<endl;
      _transformation_to_reference_position =av_motion.inverse();    
    }

  move_from_scanner = ro3d_ptr->get_transformation_from_scanner_coords();
  /* 966 transformation:
    RigidObject3DTransformation(Quaternion<float>(0.00525584F, -0.999977F, -0.00166456F, 0.0039961F),
                               CartesianCoordinate3D<float>( -1981.93F, 3.96638F, 20.1226F));
  */
  move_to_scanner = ro3d_ptr->get_transformation_to_scanner_coords();

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
process_new_time_event(const CListTime& time_event)
{
  assert(fabs(current_time - time_event.get_time_in_secs())<.0001);     
  ro3d_ptr->get_motion_rel_time(ro3dtrans,current_time);

  ro3dtrans = compose(move_to_scanner,
		      compose(_transformation_to_reference_position,
			      compose(ro3dtrans,move_from_scanner)));

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
  
  // find cartesian coordinates on LOR
  CartesianCoordinate3D<float> coord_1;
  CartesianCoordinate3D<float> coord_2;

  record.get_uncompressed_proj_data_info_sptr()->
    find_cartesian_coordinates_of_detection(coord_1,coord_2, bin);
  
  // now do the movement
#if 1
 
  const CartesianCoordinate3D<float> coord_1_transformed = ro3dtrans.transform_point(coord_1);
  const CartesianCoordinate3D<float> coord_2_transformed = ro3dtrans.transform_point(coord_2);
 
  
#else  

 const CartesianCoordinate3D<float> coord_1_transformed =
     move_to_scanner.
     transform_point(ro3d_move_to_reference_position.
                   transform_point(
                                  ro3dtrans.
                                   transform_point(move_from_scanner.
                                                   transform_point(coord_1))));
  const CartesianCoordinate3D<float> coord_2_transformed =
     move_to_scanner.
     transform_point(ro3d_move_to_reference_position.
                   transform_point(
                                  ro3dtrans.
                                   transform_point(move_from_scanner.
                                                   transform_point(coord_2))));

#endif
  proj_data_info.
    find_bin_given_cartesian_coordinates_of_detection(bin,
                                                      coord_1_transformed,
					              coord_2_transformed);

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
