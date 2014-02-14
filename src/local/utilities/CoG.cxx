//
//
/*!
  \file 
  \ingroup utilities
 
  \brief This program computes the centre of gravity in each plane and
  writes its coordinates to file, together with a weight for each plane.
  
  \par Usage
  \code
   CoG output_filename_prefix image_input_filename
  \endcode
  Output will be in files \c output_filename_prefix.x and 
  \c output_filename_prefix.y.
  Each will contain a list of all z coordinates (in mm),
  then all x/y coordinates (in mm), then all weights.

  \par Details
  The result can be used to find the central line of an (uniform) object, for
  instance a cylinder. The output of this program can by used by
  do_linear_regression.

  The weight is currently simply the sum of the voxel values in that plane,
  thresholded to be at least 0. If the weight is 0, the x,y coordinates are
  simply set to 0.

  All coordinates are in mm and in the standard STIR coordinate system,
  except that the origin in z is shifted to the centre of the planes.

  \author Kris Thielemans
*/
/*
    Copyright (C) 2000- 2007, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/


#include "stir/DiscretisedDensity.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/shared_ptr.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/CartesianCoordinate2D.h"
#include "stir/centre_of_gravity.h"
#include <iostream>
#include <iomanip>
#include <numeric>
#include <fstream>
#include <algorithm>
#include <string>

#ifndef STIR_NO_NAMESPACES
using std::ofstream;
using std::cout;
using std::setw;
using std::cerr;
using std::endl;
using std::min;
using std::max;
using std::string;
#endif

int
main(int argc, char *argv[])
{
  USING_NAMESPACE_STIR;
  if (argc!=3)
   {
      cerr <<"Usage: " << argv[0] << "  output_filename_prefix image_input_filename\n";
      cerr <<"output will be in files output_filename_prefix.x, output_filename_prefix.y\n"
	   << "Each will contain\n"
	"\ta list of all z coordinates (in mm),\n"
	"\tthen all x/y coordinates (in mm),\n"
	"\tthen all weights\n";
      return (EXIT_FAILURE);
   }

  const string output_filename_prefix = argv[1];
  const shared_ptr<DiscretisedDensity<3,float> > 
    density_ptr =  DiscretisedDensity<3,float>::read_from_file(argv[2]);
  VoxelsOnCartesianGrid<float> * const image_ptr =
    dynamic_cast<VoxelsOnCartesianGrid<float> * const>(density_ptr.get());
  if (image_ptr == 0)
    {
      warning("Can only handle images of type VoxelsOnCartesianGrid\n");
      exit(EXIT_FAILURE);
    }

  VectorWithOffset< CartesianCoordinate3D<float> > allCoG;
  VectorWithOffset<float> weights;
  find_centre_of_gravity_in_mm_per_plane(allCoG,
				   weights,
				   *image_ptr);
  {
    const string filename = output_filename_prefix + ".x";
    ofstream xout(filename.c_str());
    xout << image_ptr->get_length() << "\n";
    for (int z=image_ptr->get_min_index(); z<=image_ptr->get_max_index(); z++)
      xout << allCoG[z].z()<< "\n";
    for (int z=image_ptr->get_min_index(); z<=image_ptr->get_max_index(); z++)
      xout << allCoG[z].x() << "\n";
    for (int z=image_ptr->get_min_index(); z<=image_ptr->get_max_index(); z++)
    xout << weights[z] << "\n";
  }

  {
    const string filename = output_filename_prefix + ".y";
    ofstream yout(filename.c_str());

    yout << image_ptr->get_length() << "\n";
    for (int z=image_ptr->get_min_index(); z<=image_ptr->get_max_index(); z++)
      yout << allCoG[z].z() << "\n";
    for (int z=image_ptr->get_min_index(); z<=image_ptr->get_max_index(); z++)
      yout << allCoG[z].y() << "\n";
    for (int z=image_ptr->get_min_index(); z<=image_ptr->get_max_index(); z++)
      yout << weights[z] << "\n";
  }
  

  cout << "output written in \""
       <<output_filename_prefix + ".x"
       << "\" and \""
       << output_filename_prefix + ".y" << '\"'
       << endl;
  return EXIT_SUCCESS;

}
