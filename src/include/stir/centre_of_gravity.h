//
// $Id$
//
/*!
  \file 
  \ingroup Array
 
  \brief This file contains functions to compute the centre of gravity of
  arrays and images.
  \author Kris Thielemans
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/


#include "stir/common.h"

START_NAMESPACE_STIR
// predeclerations to avoid having to include the files and create unnecessary
// dependencies
template <int num_dimensions, class T> class BasicCoordinate;
template <int num_dimensions, class T> class Array;
template <class elemT> class VectorWithOffset;
template <class coordT> class CartesianCoordinate3D;
template <class elemT> class VoxelsOnCartesianGrid;


//! Compute centre of gravity of an Array but without dividing by its sum
/*! \ingroup Array
   Each coordinate of the unweighted centre of gravity is computed as follows:
   \f[
      C_k = \sum_{i_1...i_n} i_k A_{i1...i_n}
   \f]
*/
#ifdef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
#define T float
#define num_dimensions 2
#else
template <int num_dimensions, class T>
#endif
BasicCoordinate<num_dimensions,T> 
find_unweighted_centre_of_gravity(const Array<num_dimensions,T>& );

//! Compute centre of gravity of a 1D Array but without dividing by its sum
/*! \ingroup Array
*/   
#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
template <class T>
#endif
T 
find_unweighted_centre_of_gravity(const Array<1,T>& );

#ifdef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
#undef T
#undef num_dimensions
#endif

//! Compute centre of gravity of an Array
/*! \ingroup Array
    Calls find_unweighted_centre_of_gravity and divides the result with the
    sum of all the elements in the array.

    \warning When the sum is 0, error() is called.
    \todo better error handling
*/
template <int num_dimensions, class T>
BasicCoordinate<num_dimensions,T> 
find_centre_of_gravity(const Array<num_dimensions,T>& );

//! Computes centre of gravity for each plane
/*! \ingroup Array
  The result is in mm, taking the origin (and min/max indices) into account.
  Results are in standard STIR coordinate system.

  The result can be used to find the central line of a (uniform) object, for
  instance a cylinder. The output of this program can by used by
  do_linear_regression.

  The weight is currently simply the sum of the voxel values in that plane,
  thresholded to be at least 0. If the weight is 0, the x,y coordinates are
  simply set to 0.
 */
template <class T>
void
find_centre_of_gravity_in_mm_per_plane(  VectorWithOffset< CartesianCoordinate3D<float> >& allCoG,
					 VectorWithOffset<T>& weights,
					 const VoxelsOnCartesianGrid<T>& image);

END_NAMESPACE_STIR
