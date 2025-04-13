//
//
/*
    Copyright (C) 2003- 2007, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup Array

  \brief This file contains functions to compute the centre of gravity of
  arrays and images.
  \author Kris Thielemans
*/

#include "stir/ArrayFwd.h"

START_NAMESPACE_STIR
// predeclerations to avoid having to include the files and create unnecessary
// dependencies
template <int num_dimensions, class T>
class BasicCoordinate;
template <typename elemT>
class VectorWithOffset;
template <class coordT>
class CartesianCoordinate3D;
template <class elemT>
class VoxelsOnCartesianGrid;

//! Compute centre of gravity of a vector but without dividing by its sum
/*! \ingroup Array
   The unweighted centre of gravity is computed as follows:
   \f[
      C_k = \sum_{i} i A_{i}
   \f]
*/
template <class T>
T find_unweighted_centre_of_gravity_1d(const VectorWithOffset<T>& row);

//! Compute centre of gravity of an Array but without dividing by its sum
/*! \ingroup Array
   Each coordinate of the unweighted centre of gravity is computed as follows:
   \f[
      C_k = \sum_{i_1...i_n} i_k A_{i_1...i_n}
   \f]
*/
template <int num_dimensions, class T>
BasicCoordinate<num_dimensions, T> find_unweighted_centre_of_gravity(const ArrayType<num_dimensions, T>&);

//! Compute centre of gravity of a 1D Array but without dividing by its sum
/*! \ingroup Array
  Conceptually the same as the n-dimensional version, but returns a \c T, not a
  BasicCoordinate\<1,T\>.
*/
template <class T>
T find_unweighted_centre_of_gravity(const ArrayType<1, T>&);

//! Compute centre of gravity of an Array
/*! \ingroup Array
    Calls find_unweighted_centre_of_gravity and divides the result with the
    sum of all the elements in the array.

    \warning When the sum is 0, error() is called.
    \todo better error handling
*/
template <int num_dimensions, class T>
BasicCoordinate<num_dimensions, T> find_centre_of_gravity(const ArrayType<num_dimensions, T>&);

//! Computes centre of gravity for each plane
/*! \ingroup Array
  The result is in mm in STIR physical coordinates, i.e. taking the origin into account.

  The result can be used to find the central line of a (uniform) object, for
  instance a cylinder. The output of this function can by used by
  linear_regression().

  The weight is currently simply the sum of the voxel values in that plane,
  thresholded to be at least 0. If the weight is 0, the x,y coordinates are
  simply set to 0.
 */
template <class T>
void find_centre_of_gravity_in_mm_per_plane(VectorWithOffset<CartesianCoordinate3D<float>>& allCoG,
                                            VectorWithOffset<T>& weights,
                                            const VoxelsOnCartesianGrid<T>& image);

//! Computes centre of gravity of an image
/*! \ingroup Array
  The result is in mm in STIR physical coordinates, i.e. taking the origin into account.
*/
template <class T>
CartesianCoordinate3D<float> find_centre_of_gravity_in_mm(const VoxelsOnCartesianGrid<T>& image);

END_NAMESPACE_STIR
