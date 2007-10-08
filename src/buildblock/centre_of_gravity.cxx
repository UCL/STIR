//
// $Id$
//
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for details
*/
/*!
  \file 
  \ingroup Array
 
  \brief Implementations of centre_of_gravity.h 
  \warning Only 1, 2 and 3 dimensional versions with floats are instantiated.  
  \author Kris Thielemans
  $Date$
  $Revision$
*/


#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/centre_of_gravity.h"
#include "stir/assign.h"
#include <algorithm>

#ifndef STIR_NO_NAMESPACES
using std::min;
using std::max;
#endif


START_NAMESPACE_STIR

template <class T>
T
find_unweighted_centre_of_gravity_1d(const VectorWithOffset<T>& row)
{
  T CoG;
  assign(CoG, 0);
  for (int x=row.get_min_index(); x<=row.get_max_index(); x++)
    CoG += row[x]*x;
  return CoG;
}

#ifdef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
#define T float
#else
template <class T>
#endif
T
find_unweighted_centre_of_gravity(const Array<1,T>& row)
{
  return find_unweighted_centre_of_gravity_1d(row);
}

template <int num_dimensions, class T>
BasicCoordinate<num_dimensions,T> 
find_unweighted_centre_of_gravity(const Array<num_dimensions,T>& array)
{
  if (array.size() == 0)
    return BasicCoordinate<num_dimensions,T>(0);

  /*
    Use recursion to lower dimensional case, based on the following
    sum_ijk {i,j,k} a_ijk
    = sum_i {i,0,0} sum_jk a_ijk + 
      sum_i sum_jk {0,j,k} a_ijk
    The first term can be computed as a 1D CoG calculation, the
    last term is a sum of num_dimensions-1 CoG's.
  */
  // last term
  BasicCoordinate<num_dimensions-1,T> lower_dimension_CoG(0);
  for (int i=array.get_min_index(); i<=array.get_max_index(); ++i)
    {
      lower_dimension_CoG += find_unweighted_centre_of_gravity(array[i]);
    }

  // first term
  Array<1,T>
    first_dim_sums(array.get_min_index(), array.get_max_index());
  for (int i=array.get_min_index(); i<=array.get_max_index(); ++i)
    {
      first_dim_sums[i] = array[i].sum();
    }
  const T first_dim_CoG =
    find_unweighted_centre_of_gravity(first_dim_sums);

  // put them into 1 coordinate and return
  return join(first_dim_CoG, lower_dimension_CoG);
}


template <int num_dimensions, class T>
BasicCoordinate<num_dimensions,T> 
find_centre_of_gravity(const Array<num_dimensions,T>& array)
{
  const T sum = array.sum();

  if (sum == 0)
    error("Warning: find_centre_of_gravity cannot properly normalise, as data sum to 0\n");
  return 
    find_unweighted_centre_of_gravity(array) / sum;
}


template <class T>
void
find_centre_of_gravity_in_mm_per_plane(  VectorWithOffset< CartesianCoordinate3D<float> >& allCoG,
					 VectorWithOffset<T>& weights,
					 const VoxelsOnCartesianGrid<T>& image)
{

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
    allCoG[z] = image.get_physical_coordinates_for_indices(allCoG[z]);
  }
}

template <class T>
CartesianCoordinate3D<float>
find_centre_of_gravity_in_mm(const VoxelsOnCartesianGrid<T>& image)
{
  const BasicCoordinate<3,T> CoG = find_centre_of_gravity(image);
  return image.get_physical_coordinates_for_indices(CoG);   
}



//******* INSTANTIATIONS

// next instantiations already does 1 and 2 dimensional versions of the other functions
template 
void
find_centre_of_gravity_in_mm_per_plane(  VectorWithOffset< CartesianCoordinate3D<float> >& allCoG,
					 VectorWithOffset<float>& weights,
					 const VoxelsOnCartesianGrid<float>& image);

/*
template
BasicCoordinate<3,float>
find_centre_of_gravity(const Array<3,float>&);
*/
// this instantiates 3D versions
template 
CartesianCoordinate3D<float>
find_centre_of_gravity_in_mm(const VoxelsOnCartesianGrid<float>& image);

END_NAMESPACE_STIR
