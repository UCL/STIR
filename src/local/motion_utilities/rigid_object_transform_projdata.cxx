//
// $Id$
//
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/
namespace stir { // for doxygen
/*!
  \file
  \ingroup motion_utilities
  \brief A utility to perform a rigid object transformation on projection data.

  Basic program for moving projection data given 1 rigid object transformation,
  specified by 1 quaternion and 1 translation vector. Conventions for these are
  as for Polaris.

  \see transform_3d_object(ProjData& out_proj_data,
		    const ProjData& in_proj_data,
		    const RigidObject3DTransformation& rigid_object_transformation)

  \par Usage
  Run to get a usage message

  \author Kris Thielemans

  $Date$
  $Revision$
*/
} // END_NAMESPACE_STIR

#include "stir/ProjDataInterfile.h"
#include "stir/Succeeded.h"
#include "stir/CartesianCoordinate3D.h"
#include "local/stir/motion/RigidObject3DTransformation.h"
#include "local/stir/motion/transform_3d_object.h"
#include "local/stir/Quaternion.h"
#include "stir/CPUTimer.h"
#include <string>

USING_NAMESPACE_STIR


int main(int argc, char **argv)
{
  if (argc < 10 || argc > 13)
    {
      std::cerr << "Usage:\n"
	   << argv[0] << " output_filename input_projdata_name q0 qx qy qz tx ty tz [max_in_segment_num_to_process [template_projdata_name [max_out_segment_num_to_process ]]]\n"
	   << "max_in_segment_num_to_process defaults to all segments\n"
		<< "max_out_segment_num_to_process defaults to all segments in template\n";
      exit(EXIT_FAILURE);
    }
  const std::string  output_filename = argv[1];
  shared_ptr<ProjData> in_projdata_ptr = ProjData::read_from_file(argv[2]);  
  //const float angle_around_x =  atof(argv[3]) *_PI/180;
  Quaternion<float> quat(static_cast<float>(atof(argv[3])),
			 static_cast<float>(atof(argv[4])),
			 static_cast<float>(atof(argv[5])),
			 static_cast<float>(atof(argv[6])));
  quat.normalise();
  const CartesianCoordinate3D<float> translation(static_cast<float>(atof(argv[9])),
						 static_cast<float>(atof(argv[8])),
						 static_cast<float>(atof(argv[7])));
  const int max_in_segment_num_to_process = argc <=10 ? in_projdata_ptr->get_max_segment_num() : atoi(argv[10]);

  shared_ptr<ProjDataInfo> proj_data_info_ptr; // template for output
  int max_out_segment_num_to_process=-1;
  if (argc>11)
    {
      shared_ptr<ProjData> template_proj_data_sptr = 
	ProjData::read_from_file(argv[11]);
      proj_data_info_ptr =
	template_proj_data_sptr->get_proj_data_info_ptr()->clone();
      if (argc>12)
	max_out_segment_num_to_process = atoi(argv[12]);
    }
  else
    {
      proj_data_info_ptr =
	in_projdata_ptr->get_proj_data_info_ptr()->clone();
    }
  if (max_out_segment_num_to_process<0)
    max_out_segment_num_to_process = 
      proj_data_info_ptr->get_max_segment_num();
  else
    proj_data_info_ptr->reduce_segment_range(-max_out_segment_num_to_process,max_out_segment_num_to_process);

  ProjDataInterfile out_projdata(proj_data_info_ptr, output_filename, ios::out); 

  CPUTimer timer;
  timer.start();
  Succeeded succes =
    transform_3d_object(out_projdata, *in_projdata_ptr,
			RigidObject3DTransformation(quat, translation),
			-max_in_segment_num_to_process,
			max_in_segment_num_to_process);
  timer.stop();
  std::cerr << "CPU time " << timer.value() << '\n';
  return succes == Succeeded::yes ? EXIT_SUCCESS : EXIT_FAILURE;
}
