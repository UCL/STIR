//
//
/*!
  \file 
  \ingroup utilities

  \author Kris Thielemans
*/
/*
    Copyright (C) 2003- 2004, Hammersmith Imanet
    See STIR/LICENSE.txt for details
*/

#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/shared_ptr.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/centre_of_gravity.h"
#include "stir/linear_regression.h"
#include "stir/cross_product.h"
#include "stir/stream.h"
#include <iostream>
#include <numeric>
#include <fstream>
#include <algorithm>
#include <string>

#ifndef STIR_NO_NAMESPACES
using std::ofstream;
using std::cout;
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
  if (argc!=5)
   {
      cerr <<"Usage: " << argv[0] << "  output_filename_prefix image_input_filename radius length\n";
      cerr <<"output will be in files output_filename_prefix.par\n";
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
  const float cylinder_radius = static_cast<float>(atof(argv[3]));
  const float cylinder_length = static_cast<float>(atof(argv[4]));

  VectorWithOffset< CartesianCoordinate3D<float> > allCoG;
  VectorWithOffset<float> weights;
  find_centre_of_gravity_in_mm_per_plane(allCoG,
				   weights,
				   *image_ptr);
  float cst_x=0, scale_x=0, cst_y=0, scale_y=0;
  float chi_square, variance_of_constant, variance_of_scale, covariance_of_constant_with_scale;

  VectorWithOffset<float> CoG_x(allCoG.get_min_index(), allCoG.get_max_index());
  VectorWithOffset<float> CoG_y(allCoG.get_min_index(), allCoG.get_max_index());
  VectorWithOffset<float> CoG_z(allCoG.get_min_index(), allCoG.get_max_index());

  for (int z=allCoG.get_min_index(); z<=allCoG.get_max_index(); ++z)
    {
      CoG_x[z] = allCoG[z].x();
      CoG_y[z] = allCoG[z].y();
      CoG_z[z] = allCoG[z].z();
    }
  linear_regression(cst_x, scale_x,
		       chi_square, variance_of_constant, variance_of_scale,
		       covariance_of_constant_with_scale,
		       CoG_x,
		       CoG_z,
		       weights);
  cout << "scale_x = " << scale_x << " +- " << sqrt(variance_of_scale)
       << ", cst_x = " << cst_x << " +- " << sqrt(variance_of_constant)
       << "\nchi_square = " << chi_square
       << "\ncovariance = " << covariance_of_constant_with_scale
       << endl;

  linear_regression(cst_y, scale_y,
		       chi_square, variance_of_constant, variance_of_scale,
		       covariance_of_constant_with_scale,
		       CoG_y,
		       CoG_z,
		       weights);
  cout << "scale_y = " << scale_y << " +- " << sqrt(variance_of_scale)
       << ", cst_y = " << cst_y << " +- " << sqrt(variance_of_constant)
       << "\nchi_square = " << chi_square
       << "\ncovariance = " << covariance_of_constant_with_scale
       << endl;
  /*
    Line is given as
    x = ax z + bx
    y = ay z + by
    
    so,
    dir_z = {1,ay,ax}/norm
    dir_y and dir_x are construct such that they form a left-handed coordinate system with dir_z
  */
  CartesianCoordinate3D<float> dir_z(1,scale_y, scale_x);

  // shift origin to somewhere roughly in the middle of the image
  const CartesianCoordinate3D<float> cylinder_origin = 
    CartesianCoordinate3D<float>(0, cst_y, cst_x) + 
    dir_z * (image_ptr->get_z_size() * image_ptr->get_voxel_size().z());

  dir_z /= static_cast<float>(norm(dir_z));
  CartesianCoordinate3D<float> dir_y(-scale_y, 1, 0);
  dir_y /= static_cast<float>(norm(dir_y));
  // TODO sign of dir_x
  const CartesianCoordinate3D<float> dir_x = cross_product(dir_z, dir_y);

  const string parfile_name = output_filename_prefix + ".par";
  ofstream parfile(parfile_name.c_str());
  parfile << "generate_image Parameters :=\n"
	  << "output filename:=" << output_filename_prefix << '\n'
	  << "X output image size (in pixels):=" << image_ptr->get_x_size() << '\n'
	  << "Y output image size (in pixels):=" << image_ptr->get_y_size() << '\n'
	  << "Z output image size (in pixels):=" << image_ptr->get_z_size() << '\n'
	  << "X voxel size (in mm):= " << image_ptr->get_voxel_size().x() << '\n'
	  << "Y voxel size (in mm):= " << image_ptr->get_voxel_size().y() << '\n'
	  << "Z voxel size (in mm) :=" << image_ptr->get_voxel_size().z() << '\n';
  parfile << "shape type:= ellipsoidal cylinder\n"
	  << "Ellipsoidal Cylinder Parameters:=\n"
	  << "   radius-x (in mm):=" << cylinder_radius << '\n'
	  << "   radius-y (in mm):=" << cylinder_radius << '\n'
	  << "   length-z (in mm):=" << cylinder_length << '\n'
	  << "   origin-x (in mm):=" << cylinder_origin.x() << '\n'
	  << "   origin-y (in mm):=" << cylinder_origin.y() << '\n'
	  << "   origin-z (in mm):=" << cylinder_origin.z() << '\n'
	  << "   direction-x (in mm):=" << dir_x << '\n'
	  << "   direction-y (in mm):=" << dir_y << '\n'
	  << "   direction-z (in mm):=" << dir_z << '\n'
	  << "   END:=\n"
	  << "value :=" << 1 << '\n'
	  << "END:=\n";

  if (!parfile)
    warning("Error writing %s\n", parfile_name.c_str());
  return (parfile ? EXIT_SUCCESS : EXIT_FAILURE);

}
