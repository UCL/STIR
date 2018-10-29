//
//
/*
    Copyright (C) 2003- 2012, Hammersmith Imanet Ltd
    Internal GE use only.
*/

namespace stir { // for doxygen
/*!
  \file
  \ingroup motion_utilities
  \brief A utility to re-interpolate an image to a new coordinate system.

  Basic program for moving an image given 1 rigid object transformation,
  specified by 1 quaternion and 1 translation vector. Conventions for these are
  as for Polaris.

  \see transform_3d_object(DiscretisedDensity<3,float>& out_density, 
		    const DiscretisedDensity<3,float>& in_density, 
		    const RigidObject3DTransformation& rigid_object_transformation)

  \par Usage
  Run to get a usage message

  \author Kris Thielemans

*/
} // end namespace stir

#include "stir/DiscretisedDensity.h"
#include "stir/IO/OutputFileFormat.h"
#include "stir/IO/read_from_file.h"
#include "stir/Succeeded.h"
#include "stir/KeyParser.h"
#include "stir/is_null_ptr.h"
#include "stir_experimental/motion/RigidObject3DTransformation.h"
#include "stir_experimental/motion/transform_3d_object.h"
#include "stir_experimental/numerics/more_interpolators.h"
#include "stir_experimental/Quaternion.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/CPUTimer.h"
#include <string>

#ifndef STIR_NO_NAMESPACES
using std::string;
using std::cerr;
using std::endl;
#endif

START_NAMESPACE_STIR


#if 0

class TF
{
public:
  TF(DiscretisedDensity<3,float>& out_density,
     const DiscretisedDensity<3,float>& in_density,
     const RigidObject3DTransformation& ro_transformation)
    : out_proj_data_info_ptr(out_proj_data_info_ptr),
      in_proj_data_info_ptr(in_proj_data_info_ptr),
      ro_transformation(ro_transformation)
  {

  const VoxelsOnCartesianGrid<float> * image_ptr =
    dynamic_cast<VoxelsOnCartesianGrid<float> *>(&in_density);

  if (image_ptr==NULL)
    error("Image is not of VoxelsOnCartesianGrid type. Sorry\n");
  }

  
private:
  DiscretisedDensity<3,float>& out_density;
  RigidObject3DTransformation ro_transformation;
};

#endif

END_NAMESPACE_STIR

USING_NAMESPACE_STIR
int main(int argc, char **argv)
{
  const char * const program_name = argv[0];
  // skip program name
  --argc;
  ++argv;
  shared_ptr<OutputFileFormat<DiscretisedDensity<3,float> > >
    output_format_sptr =
    OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr();
  int interpolation_order = 1;
  bool do_origin_shift = true;

  while (argc>0 && argv[0][0]=='-')
  {
    if (strcmp(argv[0], "--output-format")==0)
      {
	if (argc<2)
	  { 
	    cerr << "Option '--output-format' expects a (filename) argument\n"; 
	    exit(EXIT_FAILURE); 
	  }
	KeyParser parser;
	parser.add_start_key("output file format parameters");
	parser.add_parsing_key("output file format type", &output_format_sptr);
	parser.add_stop_key("END"); 
	if (parser.parse(argv[1]) == false || is_null_ptr(output_format_sptr))
	  {
	    cerr << "Error parsing output file format from " << argv[1]<<endl;
	    exit(EXIT_FAILURE); 
	  }
	argc-=2; argv+=2;
    } 
    else if (strcmp(argv[0], "--interpolation_order")==0)
    {
      interpolation_order = atoi(argv[1]);
      argc-=2; argv+=2;
    }
    else if (strcmp(argv[0], "--no_origin_shift")==0)
    {
      do_origin_shift = false;
      argc-=1; argv+=1;
    }
    else
    { cerr << "Unknown option '" << argv[0] <<"'\n"; exit(EXIT_FAILURE); }
  }

  if (argc != 3)
    {
      cerr << "Usage:\n"
	   << program_name
	   << "\n  [--no_origin_shift] [--output-format parameter-filename ] [--interpolation_order 0|1]\\\n"
	   << "output_filename input_filename \"{{q0, qz, qy, qx},{ tz, ty, tx}}\"\n"
	   << "interpolation_order defaults to 1\n";
      exit(EXIT_FAILURE);
    }
  const string  output_filename = argv[0];
  const string  input_filename = argv[1];
  shared_ptr<DiscretisedDensity<3,float> >  in_density_sptr
    (read_from_file<DiscretisedDensity<3,float> >(input_filename));
  shared_ptr< DiscretisedDensity<3,float> > out_density_sptr
    (in_density_sptr->get_empty_discretised_density());

  RigidObject3DTransformation rigid_object_transformation;
  {
    std::istringstream s(argv[2]);
    s >> rigid_object_transformation;
    if (!s)
	error("error parsing transformation");
  }

  if (do_origin_shift)
    {
      const float z_shift=
	(in_density_sptr->get_min_index()+in_density_sptr->get_max_index())/2.F*
	dynamic_cast<VoxelsOnCartesianGrid<float> const&>(*in_density_sptr).get_voxel_size().z();

      RigidObject3DTransformation from_centre_to_out(Quaternion<float>(1,0,0,0),
						     CartesianCoordinate3D<float>(-z_shift,0,0));
      RigidObject3DTransformation from_in_to_centre(Quaternion<float>(1,0,0,0),
						 CartesianCoordinate3D<float>(z_shift,0,0));
      rigid_object_transformation = 
	compose(from_centre_to_out,
		compose(rigid_object_transformation, from_in_to_centre));
      std::cout << "\nTransformation after shift: " << rigid_object_transformation;
    }

  CPUTimer timer;
  timer.start();

  Succeeded success = Succeeded::yes;

  switch (interpolation_order)
    {
    case 0:
      std::cout << "Using nearest neighbour interpolation\n";
      success =     
	transform_3d_object_pull_interpolation(*out_density_sptr,
					       *in_density_sptr,
					       rigid_object_transformation.inverse(),
					       PullNearestNeighbourInterpolator<float>(),
					       /*do_jacobian=*/false ); // jacobian is 1 anyway
      break;
    case 1:
      std::cout << "Using linear interpolation\n";
      success =     
	transform_3d_object_pull_interpolation(*out_density_sptr,
					       *in_density_sptr,
					       rigid_object_transformation.inverse(),
					       PullLinearInterpolator<float>(),
					       /*do_jacobian=*/false ); // jacobian is 1 anyway
      break;
    default:
      warning("Currently only interpolation_order 0 or 1");
      exit(EXIT_FAILURE);
    }


  if (success == Succeeded::no)
    {
      warning("Error transforming data\n");
      exit(EXIT_FAILURE);
    }

  timer.stop();
  cerr << "CPU time " << timer.value() << endl;
  // write it to file
  const Succeeded succes = 
    output_format_sptr->write_to_file(output_filename, *out_density_sptr);

  return succes == Succeeded::yes ? EXIT_SUCCESS : EXIT_FAILURE;
}
