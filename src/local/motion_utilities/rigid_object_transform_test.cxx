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

#include "stir/CartesianCoordinate3D.h"
#include "stir/Succeeded.h"
#include "stir/stream.h"
#include "local/stir/motion/RigidObject3DTransformation.h"
#include <string>
#include <sstream>

#ifndef STIR_NO_NAMESPACES
using std::string;
using std::cerr;
using std::endl;
#endif



USING_NAMESPACE_STIR
int main(int argc, char **argv)
{
  const char * const program_name = argv[0];
  // skip program name
  --argc;
  ++argv;

  bool do_inverse=false;
  bool do_apply=false;
  CartesianCoordinate3D<float> point;
  float z_shift=0;

  while (argc>0 && argv[0][0]=='-')
  {
    if (strcmp(argv[0], "--inverse")==0)
      {
	do_inverse=true;
	argc-=1; argv+=1;
      } 
    else if (strcmp(argv[0], "--z_shift")==0)
      {
	z_shift = atof(argv[1]);
	argc-=2; argv+=2;
      }
    else if (strcmp(argv[0], "--apply")==0)
      {
	do_apply=true;
	std::istringstream s(argv[1]);
	s >> point;
	if (!s)
	  error("error parsing point");
	argc-=2; argv+=2;
      }
    else
      { cerr << "Unknown option '" << argv[0] <<"'\n"; exit(EXIT_FAILURE); }
  }

  if (argc !=1)
    {
      cerr << "Usage:\n"
	   << program_name
	   << "\n   --inverse | [--z-shift #] --apply {z,y,x} ]\\\n"
	   << "transformation\n"
	   << "z-shift will be added before transf, and subtracted afterwards\n";
      exit(EXIT_FAILURE);
    }

  RigidObject3DTransformation transformation;
  {
    std::istringstream s(argv[0]);
    s >> transformation;
    if (!s)
	error("error parsing transformation");
  }

  if (do_inverse)
    {
      transformation = transformation.inverse();
      std::cout << transformation;
    }
  if (do_apply)
    {
      point.z() += z_shift;
      point = transformation.transform_point(point);
      point.z() -= z_shift;
      std::cout << point;
    }

  return EXIT_SUCCESS;
}
