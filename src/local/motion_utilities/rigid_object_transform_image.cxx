//
// $Id$
//
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd
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

  $Date$
  $Revision$
*/
} // end namespace stir

#include "stir/DiscretisedDensity.h"
#include "stir/IO/DefaultOutputFileFormat.h"
#include "stir/Succeeded.h"
#include "local/stir/motion/RigidObject3DTransformation.h"
#include "local/stir/motion/transform_3d_object.h"
#include "local/stir/Quaternion.h"
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
  if (argc < 10 || argc > 12)
    {
      cerr << "Usage:\n"
	   << argv[0] << " output_filename input_filename q0 qx qy qz tx ty tz\n";
      exit(EXIT_FAILURE);
    }
  const string  output_filename = argv[1];
  const string  input_filename = argv[2];
  shared_ptr<DiscretisedDensity<3,float> >  in_density_sptr = 
    DiscretisedDensity<3,float>::read_from_file(input_filename);
  shared_ptr< DiscretisedDensity<3,float> > out_density_sptr =
    in_density_sptr->get_empty_discretised_density();

  Quaternion<float> quat(static_cast<float>(atof(argv[3])),
			 static_cast<float>(atof(argv[4])),
			 static_cast<float>(atof(argv[5])),
			 static_cast<float>(atof(argv[6])));
  quat.normalise();
  const CartesianCoordinate3D<float> translation(static_cast<float>(atof(argv[9])),
						 static_cast<float>(atof(argv[8])),
						 static_cast<float>(atof(argv[7])));

  const RigidObject3DTransformation rigid_object_transformation(quat, translation);


  CPUTimer timer;
  timer.start();

  if (transform_3d_object(*out_density_sptr, *in_density_sptr,
			  rigid_object_transformation) == Succeeded::no)
    {
      warning("Error transforming data\n");
      exit(EXIT_FAILURE);
    }

  timer.stop();
  cerr << "CPU time " << timer.value() << endl;
  // write it to file
  DefaultOutputFileFormat output_file_format;
  const Succeeded succes = 
    output_file_format.write_to_file(output_filename, *out_density_sptr);

  return succes == Succeeded::yes ? EXIT_SUCCESS : EXIT_FAILURE;
}
