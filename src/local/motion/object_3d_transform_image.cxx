//
// $Id$
//
/*!

  \file
  \ingroup motion
  \brief A utility to re-interpolate an image to a new coordinate system.

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet
    See STIR/LICENSE.txt for details
*/
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/round.h"
#include "local/stir/motion/object_3d_transform_image.h"
#include "local/stir/motion/RigidObject3DTransformation.h"
#include <algorithm>

#ifndef STIR_NO_NAMESPACES
using std::min;

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


void 
object_3d_transform_image(DiscretisedDensity<3,float>& out_density, 
			  const DiscretisedDensity<3,float>& in_density, 
			  const RigidObject3DTransformation& rigid_object_transformation)
{

  const VoxelsOnCartesianGrid<float>& in_image =
    dynamic_cast<VoxelsOnCartesianGrid<float> const&>(in_density);
  VoxelsOnCartesianGrid<float>& out_image =
    dynamic_cast<VoxelsOnCartesianGrid<float>&>(out_density);

#if 0
  for (int z= in_image.get_min_index(); z<= in_image.get_max_index(); ++z)
    for (int y= in_image[z].get_min_index(); y<= in_image[z].get_max_index(); ++y)
      for (int x= in_image[z][y].get_min_index(); x<= in_image[z][y].get_max_index(); ++x)
      {
        const CartesianCoordinate3D<float> current_point =
          CartesianCoordinate3D<float>(z,y,x) * in_image.get_voxel_size() +
          in_image.get_origin();
        const CartesianCoordinate3D<float> new_point =
          rigid_object_transformation.transform_point(current_point);
        const CartesianCoordinate3D<float> new_point_in_image_coords =
           (new_point - out_image.get_origin()) / out_image.get_voxel_size();
        // now find nearest neighbour
        const CartesianCoordinate3D<int> 
           left_neighbour(round(floor(new_point_in_image_coords.z())),
                             floor(round(new_point_in_image_coords.y())),
                             floor(round(new_point_in_image_coords.x())));

        if (left_neighbour[1] <= out_image.get_max_index() &&
            left_neighbour[1] >= out_image.get_min_index() &&
            left_neighbour[2] <= out_image[left_neighbour[1]].get_max_index() &&
            left_neighbour[2] >= out_image[left_neighbour[1]].get_min_index() &&
            left_neighbour[3] <= out_image[left_neighbour[1]][left_neighbour[2]].get_max_index() &&
            left_neighbour[3] >= out_image[left_neighbour[1]][left_neighbour[2]].get_min_index())
          out_image[left_neighbour[1]][left_neighbour[2]][left_neighbour[3]] +=
            in_image[z][y][x];
      }
#else

  const RigidObject3DTransformation inverse_rigid_object_transformation =
    rigid_object_transformation.inverse();

  for (int z= out_image.get_min_index(); z<= out_image.get_max_index(); ++z)
    for (int y= out_image[z].get_min_index(); y<= out_image[z].get_max_index(); ++y)
      for (int x= out_image[z][y].get_min_index(); x<= out_image[z][y].get_max_index(); ++x)
      {
        const CartesianCoordinate3D<float> current_point =
          CartesianCoordinate3D<float>(z,y,x) * out_image.get_voxel_size() +
          out_image.get_origin();
        const CartesianCoordinate3D<float> new_point =
          inverse_rigid_object_transformation.transform_point(current_point);
        const CartesianCoordinate3D<float> new_point_in_image_coords =
           (new_point - in_image.get_origin()) / in_image.get_voxel_size();
        // now find left neighbour
        const CartesianCoordinate3D<int> 
           left_neighbour(round(floor(new_point_in_image_coords.z())),
                             round(floor(new_point_in_image_coords.y())),
                             round(floor(new_point_in_image_coords.x())));

        if (left_neighbour[1] < in_image.get_max_index() &&
            left_neighbour[1] > in_image.get_min_index() &&
            left_neighbour[2] < in_image[left_neighbour[1]].get_max_index() &&
            left_neighbour[2] > in_image[left_neighbour[1]].get_min_index() &&
            left_neighbour[3] < in_image[left_neighbour[1]][left_neighbour[2]].get_max_index() &&
            left_neighbour[3] > in_image[left_neighbour[1]][left_neighbour[2]].get_min_index())
	  {
	    const int x1=left_neighbour[3];
	    const int y1=left_neighbour[2];
	    const int z1=left_neighbour[1];
	    const int x2=left_neighbour[3]+1;
	    const int y2=left_neighbour[2]+1;
	    const int z2=left_neighbour[1]+1;
	    const float ix = new_point_in_image_coords[3]-x1;
	    const float iy = new_point_in_image_coords[2]-y1;
	    const float iz = new_point_in_image_coords[1]-z1;
	    const float ixc = 1 - ix;
	    const float iyc = 1 - iy;
	    const float izc = 1 - iz;
	    out_image[z][y][x] =
	      ixc * (iyc * (izc * in_image[z1][y1][x1]
			    + iz  * in_image[z2][y1][x1])
		     + iy * (izc * in_image[z1][y2][x1]
			      + iz  * in_image[z2][y2][x1])) 
	      + ix * (iyc * (izc * in_image[z1][y1][x2]
			     + iz  * in_image[z2][y1][x2])
		      + iy * (izc * in_image[z1][y2][x2]
			      + iz  * in_image[z2][y2][x2]));
	  }
	else
	  out_image[z][y][x] = 0;

      }
#endif
}

END_NAMESPACE_STIR
