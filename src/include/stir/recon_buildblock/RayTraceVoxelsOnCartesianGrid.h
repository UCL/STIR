//
// $Id$
//
/*!

  \file
  \ingroup recon_buildblock
  \brief Declaration of stir::RayTraceVoxelsOnCartesianGrid

  \author Kris Thielemans
  \author PARAPET project

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
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
#include "stir/common.h"

START_NAMESPACE_STIR

class ProjMatrixElemsForOneBin;
template <typename elemT> class CartesianCoordinate3D;

/*! \ingroup recon_buildblock
  
  \brief RayTraceVoxelsOnCartesianGrid finds the Length of Intersections (LOIs)
  of an LOR with a grid of voxels and appends them to
  the ProjMatrixElemsForOneBin object.

  \param lor object to which the intersected voxels and the LOI will be appended
  \param start_point first point on the LOR. The first voxel will contain this point.
  \param end_point last point on the LOR. The last voxel will contain this point.
  \param voxel_size normally in mm
  \param normalisation_constant LOIs will be multiplied with this constant

  start_point and end_point have to be given in 'voxel grid
  units' (i.e. voxels are spaced 1 unit apart). The centre
  of the voxels are assumed to be at integer coordinates 
  (e.g. (0,0,0) is the centre of a voxel).

  For the start voxel, the intersection length of the LOR with the
  whole voxel is computed, not just from the start_point to the next edge. The
  same is true for the end voxel.

  \warning RayTraceVoxelsOnCartesianGrid appends voxels and intersection lengths to the lor.
  It does NOT reset it first.

  RayTraceVoxelsOnCartesianGrid is based on Siddon's algorithm.

  Siddon's algorithm works by looking at intersections of the 
  'intra-voxel' planes with the LOR.

  The LORs is parametrised as
  \code
  (x,y,z) = a (1/inc_x, 1/inc_y, 1/inc_z) + start_point
  \endcode
  Then values of 'a' are computed where the LOR intersects an intra-voxel plane.
  For example, 'ax' are the values where x= n + 0.5 (for integer n).
  Finally, we can go along the LOR and check which of the ax,ay,az is smallest,
  as this determines which plane the LOR intersects at this point.

*/
void 
RayTraceVoxelsOnCartesianGrid(ProjMatrixElemsForOneBin& lor, 
                              const CartesianCoordinate3D<float>& start_point, 
                              const CartesianCoordinate3D<float>& end_point, 
                              const CartesianCoordinate3D<float>& voxel_size,
                              const float normalisation_constant = 1.F);

END_NAMESPACE_STIR
