//
// $Id$
//
/*!

  \file
  \ingroup utilities
  \brief A utility to re-interpolate an image to a new coordinate system.

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2003- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/
#include "stir/DiscretisedDensity.h"
#include "stir/IO/DefaultOutputFileFormat.h"
#include "stir/Succeeded.h"
#include "local/stir/motion/RigidObject3DTransformation.h"
#include "local/stir/motion/object_3d_transform_image.h"
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

  const Quaternion<float> quat(atof(argv[3]),atof(argv[4]),atof(argv[5]),atof(argv[6]));
  const CartesianCoordinate3D<float> translation(atof(argv[9]),atof(argv[8]),atof(argv[7]));

  const RigidObject3DTransformation rigid_object_transformation(quat, translation);


  CPUTimer timer;
  timer.start();

  object_3d_transform_image(*out_density_sptr, *in_density_sptr,
			    rigid_object_transformation);

  timer.stop();
  cerr << "CPU time " << timer.value() << endl;
  // write it to file
  DefaultOutputFileFormat output_file_format;
  const Succeeded succes = 
    output_file_format.write_to_file(output_filename, *out_density_sptr);

  return succes == Succeeded::yes ? EXIT_SUCCESS : EXIT_FAILURE;
}
