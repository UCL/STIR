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
  const char * const program_name = argv[0];
  // skip program name
  --argc;
  ++argv;

  bool do_origin_shift = true;
  while (argc>0 && argv[0][0]=='-')
  {
    if (strcmp(argv[0], "--no_origin_shift")==0)
    {
      do_origin_shift = false;
      argc-=1; argv+=1;
    }
    else
      { std::cerr << "Unknown option '" << argv[0] <<"'\n"; exit(EXIT_FAILURE); }
  }

  if (argc < 3 || argc > 6)
    {
      std::cerr << "Usage:\n"
		<< program_name 
		<< "\n\t --no_origin_shift\\"
		<< "\n\t output_filename input_projdata_name  \\"
		<< "\n\t \"{{q0, qz, qy, qx},{ tz, ty, tx}}\"\\"
		<<"\n\t [max_in_segment_num_to_process [template_projdata_name [max_out_segment_num_to_process ]]]\n"
		<< "max_in_segment_num_to_process defaults to all segments\n"
		<< "max_out_segment_num_to_process defaults to all segments in template\n";
      exit(EXIT_FAILURE);
    }
  const std::string  output_filename = argv[0];
  shared_ptr<ProjData> in_projdata_ptr = ProjData::read_from_file(argv[1]);  
  //const float angle_around_x =  atof(argv[3]) *_PI/180;
  RigidObject3DTransformation rigid_object_transformation;
  {
    std::istringstream s(argv[2]);
    s >> rigid_object_transformation;
    if (!s)
	error("error parsing transformation");
  }

  const int max_in_segment_num_to_process = argc <=4 ? in_projdata_ptr->get_max_segment_num() : atoi(argv[3]);

  shared_ptr<ProjDataInfo> proj_data_info_ptr; // template for output
  int max_out_segment_num_to_process=-1;
  if (argc<=5)
    {
      shared_ptr<ProjData> template_proj_data_sptr = 
	ProjData::read_from_file(argv[4]);
      proj_data_info_ptr =
	template_proj_data_sptr->get_proj_data_info_ptr()->clone();
      if (argc<=6)
	max_out_segment_num_to_process = atoi(argv[5]);
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

  if (do_origin_shift)
    {
      const float in_z_shift =
	-in_projdata_ptr->get_proj_data_info_ptr()->get_m(Bin(0,0,0,0));
      const float out_z_shift =
	-proj_data_info_ptr->get_m(Bin(0,0,0,0));

      RigidObject3DTransformation from_centre_to_out(Quaternion<float>(1,0,0,0),
						     CartesianCoordinate3D<float>(-out_z_shift,0,0));
      RigidObject3DTransformation from_in_to_centre(Quaternion<float>(1,0,0,0),
						 CartesianCoordinate3D<float>(in_z_shift,0,0));
      rigid_object_transformation = 
	compose(from_centre_to_out,
		compose(rigid_object_transformation, from_in_to_centre));
      std::cout << "\nTransformation after shift: " << rigid_object_transformation;
    }

  CPUTimer timer;
  timer.start();
  Succeeded succes =
    transform_3d_object(out_projdata, *in_projdata_ptr,
			rigid_object_transformation,
			-max_in_segment_num_to_process,
			max_in_segment_num_to_process);
  timer.stop();
  std::cerr << "CPU time " << timer.value() << '\n';
  return succes == Succeeded::yes ? EXIT_SUCCESS : EXIT_FAILURE;
}
