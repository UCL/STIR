//
// $Id$
//
/*!
  \file 
  \ingroup Array
 
  \brief Implementations of centre_of_gravity.h 
  \warning Only 1 and 2 dimensional versions are instantiated. Also, 
  for old compilers, the element type of the vectors has to be float.
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
#include "stir/CartesianCoordinate2D.h"
#include "stir/centre_of_gravity.h"
#include <algorithm>

#ifndef STIR_NO_NAMESPACES
using std::min;
using std::max;
#endif


START_NAMESPACE_STIR

#ifdef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
#define T float
#else
template <class T>
#endif
T
find_unweighted_centre_of_gravity(const Array<1,T>& row)
{
  T CoG = 0;
  for (int x=row.get_min_index(); x<=row.get_max_index(); x++)
    CoG += x*row[x];
  return CoG;
}

#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
template <class T>
#endif
BasicCoordinate<2,T> 
find_unweighted_centre_of_gravity(const Array<2,T>& plane)
{
  BasicCoordinate<2,T> CoG;
  CoG[1] = CoG[2] = 0;

  CartesianCoordinate2D<int> min_indices, max_indices;
  if (!plane.get_regular_range(min_indices, max_indices))
    error("_centre_of_gravity can handle only square arrays\n");

  Array<1,T> sum_over_y(min_indices.x(), max_indices.x());
  Array<1,T> sum_over_x(min_indices.y(), max_indices.y());


  for (int x=min_indices.x(); x<=max_indices.x(); x++)
    for (int y=min_indices.y(); y<=max_indices.y(); y++)
    {
      sum_over_y[x] += plane[y][x];
      sum_over_x[y] += plane[y][x];
    }

  CoG[2] = find_unweighted_centre_of_gravity(sum_over_y);
  CoG[1] = find_unweighted_centre_of_gravity(sum_over_x);
  
  return CoG;
}

#ifdef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
#undef T
#endif

template <int num_dimensions, class T>
BasicCoordinate<num_dimensions,T> 
find_centre_of_gravity(const Array<num_dimensions,T>& plane)
{
  const T sum = plane.sum();

  if (sum == 0)
    error("Warning: find_centre_of_gravity cannot properly normalise, as data sum to 0\n");
  return 
    find_unweighted_centre_of_gravity(plane) / sum;
}


template <class T>
void
find_centre_of_gravity_in_mm_per_plane(  VectorWithOffset< CartesianCoordinate3D<float> >& allCoG,
					 VectorWithOffset<T>& weights,
					 const VoxelsOnCartesianGrid<T>& image)
{
  const CartesianCoordinate3D<float> voxel_size = image.get_voxel_size();
  const float centre_z = 
    (image.get_min_index()+image.get_max_index())*voxel_size.z()/2;
  const CartesianCoordinate3D<float> origin = 
    image.get_origin() - CartesianCoordinate3D<float>(centre_z,0.F,0.F);

  allCoG = 
    VectorWithOffset< CartesianCoordinate3D<float> > 
    (image.get_min_index(), image.get_max_index());
  weights =
    VectorWithOffset<T> 
    (image.get_min_index(), image.get_max_index());

  for (int z=image.get_min_index(); z<=image.get_max_index(); z++)
  {
    weights[z] = max(image[z].sum(), 0.F);
    if (weights[z]==0)
      allCoG[z] = CartesianCoordinate3D<float>(0.F,0.F,0.F);
    else
      {
	const BasicCoordinate<2,T> CoG = find_centre_of_gravity(image[z]);
	allCoG[z].y() = CoG[1];
	allCoG[z].x() = CoG[2];
      }
    allCoG[z].z() = static_cast<float>(z);
    allCoG[z] *= voxel_size;
    allCoG[z] += origin;
  }
}



//******* INSTANTIATIONS
/*
template <int num_dimensions, class T>
BasicCoordinate<num_dimensions,T> 
find_centre_of_gravity(const Array<num_dimensions,T>& plane);
*/

// next instantiations already does 2 dimensional versions of the other functions
template 
void
find_centre_of_gravity_in_mm_per_plane(  VectorWithOffset< CartesianCoordinate3D<float> >& allCoG,
					 VectorWithOffset<float>& weights,
					 const VoxelsOnCartesianGrid<float>& image);

END_NAMESPACE_STIR
