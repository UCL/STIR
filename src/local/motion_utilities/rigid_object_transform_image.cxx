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
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/IO/DefaultOutputFileFormat.h"
#include "stir/Succeeded.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/round.h"
#include "local/stir/motion/RigidObject3DTransformation.h"
#include "local/stir/Quaternion.h"
#include "stir/CPUTimer.h"
#include <string>
#include <algorithm>

#ifndef STIR_NO_NAMESPACES
using std::string;
using std::cerr;
using std::endl;
using std::min;
using std::max;
#endif

START_NAMESPACE_STIR


#if 0
#define VALUE(p,x,y,z) *(p[z] + (y)*sizeX + (x))

PlaneEl_t Interpolate_3d(p,P_pt,sizeX,sizeY,sizeZ)
PlaneEl_t    *p[];
int sizeX,sizeY,sizeZ;
point *P_pt;
{   double ix,iy,iz,ixc,iyc,izc,temp;
    int   x1,x2,y1,y2,z1,z2;

    /* coordinates in x,y,z are 1 unit apart */

    if ((P_pt->x<(coord)0)|| (P_pt->y<(coord)0)|| (P_pt->z<(coord)0))
      return (PlaneEl_t)0;
    x2 = (x1 = (int)P_pt->x) + 1;
    y2 = (y1 = (int)P_pt->y) + 1;
    z2 = (z1 = (int)P_pt->z) + 1;
    if ((x2>sizeX-1)  || (y2>sizeY-1)  || (z2>sizeZ-1))
      return (PlaneEl_t)0;
    ixc = 1 - (ix = P_pt->x - x1);
    iyc = 1 - (iy = P_pt->y - y1);
    izc = 1 - (iz = P_pt->z - z1);
    temp = ixc * (iyc * (izc * VALUE(p,x1,y1,z1)
                       + iz  * VALUE(p,x1,y1,z2))
                 + iy * (izc * VALUE(p,x1,y2,z1)
                       + iz  * VALUE(p,x1,y2,z2)));
    return (PlaneEl_t) (temp
          + ix * (iyc * (izc * VALUE(p,x2,y1,z1)
                       + iz  * VALUE(p,x2,y1,z2))
                 + iy * (izc * VALUE(p,x2,y2,z1)
                       + iz  * VALUE(p,x2,y2,z2))));
}
#endif

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
  shared_ptr<DiscretisedDensity<3,float> >  in_density_ptr = 
    DiscretisedDensity<3,float>::read_from_file(input_filename);

  const Quaternion<float> quat(atof(argv[3]),atof(argv[4]),atof(argv[5]),atof(argv[6]));
  const CartesianCoordinate3D<float> translation(atof(argv[9]),atof(argv[8]),atof(argv[7]));

  RigidObject3DTransformation rigid_object_transformation(quat, translation);

  VoxelsOnCartesianGrid<float>& in_image =
    dynamic_cast<VoxelsOnCartesianGrid<float>&>(*in_density_ptr);
  shared_ptr< VoxelsOnCartesianGrid<float> > out_image_sptr =
    in_image.get_empty_voxels_on_cartesian_grid();

  VoxelsOnCartesianGrid<float>& out_image = *out_image_sptr;

  CPUTimer timer;
  timer.start();
  for (int z= in_image.get_min_index(); z<= in_image.get_max_index(); ++z)
    for (int y= in_image[z].get_min_index(); y<= in_image[z].get_max_index(); ++y)
      for (int x= in_image[z][y].get_min_index(); x<= in_image[z][y].get_max_index(); ++x)
      {
        const CartesianCoordinate3D<float> current_point =
          CartesianCoordinate3D<float>(z,y,x) * in_image.get_voxel_size() +
          in_image.get_origin();
#if 1
        // go to Polaris
        const CartesianCoordinate3D<float> 
          current_point_Polaris(-current_point.z(),
                                -current_point.x(),
                                -current_point.y());
        const CartesianCoordinate3D<float> new_point_Polaris =
          rigid_object_transformation.transform_point(current_point_Polaris);
        const CartesianCoordinate3D<float> 
          new_point(-new_point_Polaris.z(),
                    -new_point_Polaris.x(),
                    -new_point_Polaris.y());

#else
        const CartesianCoordinate3D<float> new_point =
          rigid_object_transformation.transform_point(current_point);
#endif
        const CartesianCoordinate3D<float> new_point_in_image_coords =
           (new_point - out_image.get_origin()) / out_image.get_voxel_size();
        // now find nearest neighbour
        const CartesianCoordinate3D<int> 
           nearest_neighbour(round(new_point_in_image_coords.z()),
                             round(new_point_in_image_coords.y()),
                             round(new_point_in_image_coords.x()));

        if (nearest_neighbour[1] <= out_image.get_max_index() &&
            nearest_neighbour[1] >= out_image.get_min_index() &&
            nearest_neighbour[2] <= out_image[z].get_max_index() &&
            nearest_neighbour[2] >= out_image[z].get_min_index() &&
            nearest_neighbour[3] <= out_image[z][y].get_max_index() &&
            nearest_neighbour[3] >= out_image[z][y].get_min_index())
          out_image[nearest_neighbour[1]][nearest_neighbour[2]][nearest_neighbour[3]] +=
            in_image[z][y][x];
      }

  timer.stop();
  cerr << "CPU time " << timer.value() << endl;
  // write it to file
  DefaultOutputFileFormat output_file_format;
  const Succeeded succes = 
    output_file_format.write_to_file(output_filename, out_image);

  return succes == Succeeded::yes ? EXIT_SUCCESS : EXIT_FAILURE;
}
