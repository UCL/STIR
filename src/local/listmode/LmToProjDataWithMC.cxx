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
#include "local/stir/listmode/CListRecordECAT966.h"
#include "stir/IO/stir_ecat7.h"
#include "stir/Succeeded.h"
#include <time.h>
#include "stir/is_null_ptr.h"
#include "stir/stream.h"

START_NAMESPACE_STIR

void 
LmToProjDataWithMC::set_defaults()
{
  LmToProjData::set_defaults();
  ro3d_ptr = 0; 
}

void 
LmToProjDataWithMC::initialise_keymap()
{
  LmToProjData::initialise_keymap();
  parser.add_start_key("LmToProjDataWithMC Parameters");
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

  // TODO move to RigidObject3DMotion
  if (!ro3d_ptr->is_time_offset_set())
    ro3d_ptr->synchronise(*lm_data_ptr);

  cerr << "Time offset is set to "<< ro3d_ptr->get_time_offset() << endl;
  move_from_scanner =
    RigidObject3DTransformation(Quaternion<float>(0.00525584F, -0.999977F, -0.00166456F, 0.0039961F),
                               CartesianCoordinate3D<float>( -1981.93F, 3.96638F, 20.1226F));
  move_to_scanner = move_from_scanner.inverse();

  return false;
}


void
LmToProjDataWithMC::
process_new_time_event(const CListTime& time_event)
{
  assert(fabs(current_time - time_event.get_time_in_secs())<.0001);
#if 0
  RigidObject3DTransformation ro3dtrans_1(Quaternion<float>(0.9977, 0.0148, 0.0608, -0.0237), 
    CartesianCoordinate3D<float>(-1964.4,-69.24,-37.01));
  RigidObject3DTransformation ro3dtrans_2(Quaternion<float>(0.9977, 0.0154, 0.0606, -0.024), 
    CartesianCoordinate3D<float>(-1875.15,-67.34,-36.48));
  RigidObject3DTransformation ro3dtrans_3(Quaternion<float>(0.9977,0.0138, 0.0614, -0.024), 
    CartesianCoordinate3D<float>(-1876.14,-66.21,-59.69));
  RigidObject3DTransformation ro3dtrans_4(Quaternion<float>(0.9976, 0.015, 0.0615, -0.0242), 
    CartesianCoordinate3D<float>(-1877.02,-66.46,27.43));
  RigidObject3DTransformation ro3dtrans_5 (Quaternion<float>(0.9976, 0.0145, 0.0628, \
-0.0248), 
    CartesianCoordinate3D<float>(-1935.34,-65.86,27.01));

  if (current_time >=0 && current_time <=558)
  {
    ro3dtrans= ro3dtrans_1;
  }
  else if(current_time>=573 && current_time <=906)
  {
    ro3dtrans= ro3dtrans_2;
  }
  else if (current_time>=936 && current_time <=1185)
  {
    ro3dtrans= ro3dtrans_3;
  }
  else if ( current_time>=1200 && current_time <=1527)
  {
    ro3dtrans= ro3dtrans_4;
  }
  else if (current_time>=1566 && current_time <=1860)
  {
    ro3dtrans= ro3dtrans_5;
  }
  else 
  { cerr << " STop no more frames " << endl;
  }


#else
     
  ro3d_ptr->get_motion(ro3dtrans,current_time);
#endif

         ro3dtrans = compose(move_to_scanner,
			     compose(ro3d_ptr->get_transformation_to_reference_position(),
				     compose(ro3dtrans,move_from_scanner)));

}

void 
LmToProjDataWithMC::get_bin_from_event(Bin& bin, const CListEvent& event_of_general_type) const
{
  const CListRecordECAT966& record = 
    static_cast<CListRecordECAT966 const&>(event_of_general_type);// TODO get rid of this

  const ProjDataInfoCylindricalNoArcCorr& proj_data_info =


  static_cast<const ProjDataInfoCylindricalNoArcCorr&>(*template_proj_data_info_ptr);
  const double start_time = frame_defs.get_start_time(current_frame_num);
  const double end_time = frame_defs.get_end_time(current_frame_num);

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

  // given two CartesianCoordinates find the intersection     
  if (bin.get_bin_value() > 0)
  {
    // now normalise event taking into account the
    // normalisation factor before motion correction and the number of
    // uncompressed bins that contribute to this bin
    // TODO this normalisation is not really correct
    bin.set_bin_value(1/ 
		      (bin_efficiency*
		       proj_data_info.
		         get_num_ring_pairs_for_segment_axial_pos_num(bin.segment_num(),
								    bin.axial_pos_num())*
		       proj_data_info.get_view_mashing_factor()));
  }

  
}


END_NAMESPACE_STIR
