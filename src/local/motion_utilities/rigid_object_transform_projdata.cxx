//
// $Id$
//
/*!
  \file
  \ingroup utilities
  \brief A utility to perform a rigid object transformation on projection data.

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/
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
  if (argc < 10 || argc > 12)
    {
      std::cerr << "Usage:\n"
	   << argv[0] << " output_filename input_projdata_name q0 qx qy qz tx ty tz [max_in_segment_num_to_process [max_out_segment_num_to_process ]]\n"
	   << "max_in_segment_num_to_process defaults to all segments\n"
	   << "max_out_segment_num_to_process defaults to max_in_segment_num_to_process\n";
      exit(EXIT_FAILURE);
    }
  const std::string  output_filename = argv[1];
  shared_ptr<ProjData> in_projdata_ptr = ProjData::read_from_file(argv[2]);  
  //const float angle_around_x =  atof(argv[3]) *_PI/180;
  Quaternion<float> quat(atof(argv[3]),atof(argv[4]),atof(argv[5]),atof(argv[6]));
  quat.normalise();
  const CartesianCoordinate3D<float> translation(atof(argv[9]),atof(argv[8]),atof(argv[7]));
  const int max_in_segment_num_to_process = argc <=10 ? in_projdata_ptr->get_max_segment_num() : atoi(argv[10]);
  const int max_out_segment_num_to_process = argc <=11 ? max_in_segment_num_to_process : atoi(argv[11]);


  shared_ptr<ProjDataInfo> proj_data_info_ptr =
    in_projdata_ptr->get_proj_data_info_ptr()->clone();
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
