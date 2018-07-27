//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2007, Hammersmith Imanet Ltd
    Copyright (C) 2018, University College London
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

#ifndef __stir_VoxelsOnCartesianGrid_H__
#define __stir_VoxelsOnCartesianGrid_H__

/*!
  \file 
  \ingroup densitydata 
  \brief defines the stir::VoxelsOnCartesianGrid class 

  \author Sanida Mustafovic 
  \author Kris Thielemans (with help from Alexey Zverovich)
  \author PARAPET project


*/
#include "stir/DiscretisedDensityOnCartesianGrid.h"
#include "stir/CartesianCoordinate3D.h"

START_NAMESPACE_STIR

class ProjDataInfo;
template <typename elemT> class PixelsOnCartesianGrid;

/*!
  \ingroup densitydata
  \brief This class is used to represent voxelised densities on a cuboid
  grid (3D).

  This class represents 'normal' data. Basisfunctions are just voxels.
*/
template<class elemT>
class VoxelsOnCartesianGrid:public DiscretisedDensityOnCartesianGrid<3,elemT>
{

public:

#if 0
  //! Asks for filename etc, and returns an image
static VoxelsOnCartesianGrid ask_parameters();
#endif

//! Construct an empty VoxelsOnCartesianGrid (empty range, 0 origin, 0 grid_spacing)
VoxelsOnCartesianGrid();

//! Construct a VoxelsOnCartesianGrid, initialising data from the Array<3,elemT> object.
VoxelsOnCartesianGrid(const Array<3,elemT>& v,
		      const CartesianCoordinate3D<float>& origin,
		      const BasicCoordinate<3,float>& grid_spacing);

//! Construct a VoxelsOnCartesianGrid from an index range
/*! All elements are set 0. */
VoxelsOnCartesianGrid(const IndexRange<3>& range,
		      const CartesianCoordinate3D<float>& origin, 
		      const BasicCoordinate<3,float>& grid_spacing);

 //! Construct a VoxelsOnCartesianGrid, initialising data from the Array<3,elemT> object.
VoxelsOnCartesianGrid(const shared_ptr < ExamInfo > & exam_info_sptr,
                      const Array<3,elemT>& v,
		      const CartesianCoordinate3D<float>& origin,
		      const BasicCoordinate<3,float>& grid_spacing);

//! Construct a VoxelsOnCartesianGrid from an index range
/*! All elements are set 0. */
VoxelsOnCartesianGrid(const shared_ptr < ExamInfo > & exam_info_sptr,
                      const IndexRange<3>& range,
		      const CartesianCoordinate3D<float>& origin, 
		      const BasicCoordinate<3,float>& grid_spacing);

  // KT 10/12/2001 replace 2 constructors with the more general one below
//! use ProjDataInfo to obtain the size information
/*!
   When sizes.x() is -1, a default size in x is found by taking the diameter 
   of the FOV spanned by the projection data. Similar for sizes.y().

   When sizes.z() is -1, a default size in z is found by taking the number of planes as
   <ul>
   <li> $N_0$ when segment 0 is axially compressed,</li>
   <li> $2N_0-1$ when segment 0 is not axially compressed,</li>
   </ul>
   where $N_0$ is the number of sinograms in segment 0.

   Actual index ranges start from 0 for z, but from -(x_size_used/2) for x (and similar for y).

   x,y grid spacing are set to the 
   <code>proj_data_info_ptr-\>get_scanner_ptr()-\>get_default_bin_size()/zoom</code>.
   This is to make sure that the voxel size is independent on if arc-correction is used or not.
   If the default bin size is 0, the sampling distance in s (for bin 0) is used.

   z grid spacing is set to half the scanner ring distance.

   All voxel values are set 0.

*/
VoxelsOnCartesianGrid(const ProjDataInfo& proj_data_info_ptr,
              const float zoom = 1.F);

//! Constructor from exam_info and proj_data_info
/*! \see VoxelsOnCartesianGrid(const ProjDataInfo&,
		      const float zoom,
		      const CartesianCoordinate3D<float>&, 
		      const CartesianCoordinate3D<int>& );
*/
VoxelsOnCartesianGrid(const shared_ptr < ExamInfo > & exam_info_sptr,
                      const ProjDataInfo& proj_data_info,
              const float zoom = 1.F);

//! Constructor from proj_data_info
/*! \see VoxelsOnCartesianGrid(const ProjDataInfo&,
              const float zoom,
              const CartesianCoordinate3D<float>&,
              const CartesianCoordinate3D<int>& );
*/
VoxelsOnCartesianGrid(const ProjDataInfo& proj_data_info_ptr,
              const float zoom,
              const CartesianCoordinate3D<float>& origin,
              const CartesianCoordinate3D<int>& sizes = CartesianCoordinate3D<int>(-1,-1,-1));

//! Constructor from exam_info and proj_data_info
/*! \see VoxelsOnCartesianGrid(const ProjDataInfo&,
              const float zoom,
              const CartesianCoordinate3D<float>&,
              const CartesianCoordinate3D<int>& );
*/
VoxelsOnCartesianGrid(const shared_ptr < ExamInfo > & exam_info_sptr,
                      const ProjDataInfo& proj_data_info,
              const float zoom,
              const CartesianCoordinate3D<float>& origin,
              const CartesianCoordinate3D<int>& sizes = CartesianCoordinate3D<int>(-1,-1,-1));

//! Definition of the pure virtual defined in DiscretisedDensity
#ifdef STIR_NO_COVARIANT_RETURN_TYPES
DiscretisedDensity<3,elemT>*
#else
VoxelsOnCartesianGrid<elemT>*
#endif
 get_empty_copy() const;

//! Like get_empty_copy, but returning a pointer to a VoxelsOnCartesianGrid
VoxelsOnCartesianGrid<elemT>* get_empty_voxels_on_cartesian_grid() const;

#ifdef STIR_NO_COVARIANT_RETURN_TYPES
virtual DiscretisedDensity<3,elemT>*
#else
virtual VoxelsOnCartesianGrid<elemT>*
#endif
clone() const;

//! Extract a single plane
PixelsOnCartesianGrid<elemT> get_plane(const int z) const;

//! Set a single plane
void set_plane(const PixelsOnCartesianGrid<elemT>& plane, const int z);

//! is the same as get_grid_spacing(), but now returns CartesianCoordinate3D for convenience
inline CartesianCoordinate3D<float> get_voxel_size() const;

//! is the same as set_grid_spacing()
void set_voxel_size(const BasicCoordinate<3,float>&);

//! Growing of outer dimension only
void grow_z_range(const int min_z, const int max_z);

  //! \name Convenience functions for regular grids in 3D
  /*! It is assumed that \c z is the highest dimension and \c x the lowest.
  */
  //@{
  inline int get_x_size() const;
  
  inline int get_y_size() const;
  
  inline int get_z_size() const;
  
  inline int get_min_x() const;
  
  inline int get_min_y() const;

  inline int get_min_z() const;
  
  inline int get_max_x() const;
  
  inline int get_max_y() const;
  
  inline int get_max_z() const;

  BasicCoordinate<3,int> get_lengths() const;
  BasicCoordinate<3,int> get_min_indices() const;
  BasicCoordinate<3,int> get_max_indices() const;

  //@}
protected:

  //! Set up voxels on cartesian grid (avoids code repetition, will be redundant in C++11, as constructors can call different constructors)
  void set_up_voxels_on_cartesian_grid(const ProjDataInfo& proj_data_info,
                                       const float zoom,
                                       const CartesianCoordinate3D<float>& origin,
                                       const CartesianCoordinate3D<int>& sizes);

};


END_NAMESPACE_STIR

#include "stir/VoxelsOnCartesianGrid.inl"
#endif












