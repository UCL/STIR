//
// $Id$: $Date$
//

#ifndef __VoxelsOnCartesianGrid_H__
#define __VoxelsOnCartesianGrid_H__

/*!
  \file 
  \ingroup buildblock 
  \brief defines the VoxelsOnCartesianGrid class 

  \author Sanida Mustafovic 
  \author Kris Thielemans (with help from Alexey Zverovich)
  \author PARAPET project

  \date    $Date$

  \version $Revision$

*/
#include "PixelsOnCartesianGrid.h"

START_NAMESPACE_TOMO

class ProjDataInfo;

/*!
  \ingroup buildblock
  \brief This class is used to represent voxelised densities on a cuboid
  grid (3D).

  This class represents 'normal' data. Basisfunctions are just voxels.
*/
template<class elemT>
class VoxelsOnCartesianGrid:public DiscretisedDensityOnCartesianGrid<3,elemT>
{

public:

  //! Asks for filename etc, and returns an image
static VoxelsOnCartesianGrid ask_parameters();

//! Construct an empty VoxelsOnCartesianGrid (empty range, 0 origin, 0 grid_spacing)
inline VoxelsOnCartesianGrid();

//! Construct a VoxelsOnCartesianGrid, initialising data from the Array<3,elemT> object.
inline VoxelsOnCartesianGrid(const Array<3,elemT>& v,
		      const CartesianCoordinate3D<float>& origin,
		      const BasicCoordinate<3,float>& grid_spacing);

//! Construct a VoxelsOnCartesianGrid, setting elements to 0
inline VoxelsOnCartesianGrid(const IndexRange<3>& range,
		      const CartesianCoordinate3D<float>& origin, 
		      const BasicCoordinate<3,float>& grid_spacing);

//! use PETScanInfo to obtain the size information
VoxelsOnCartesianGrid(const ProjDataInfo& proj_data_info_ptr,
		      const float zoom = 1.F, 
		      const CartesianCoordinate3D<float>& origin = CartesianCoordinate3D<float>(0.F,0.F,0.F) ,
		      const bool make_xy_size_odd = true);

//! use PETScanInfo to obtain the size information, but use non-default size for x,y
VoxelsOnCartesianGrid(const ProjDataInfo& proj_data_info_ptr,
		      const float zoom,
		      const CartesianCoordinate3D<float>& origin, 
		      const int xy_size);


//! Definition of the pure virtual defined in DiscretisedDensity
#ifdef TOMO_NO_COVARIANT_RETURN_TYPES
inline DiscretisedDensity<3,elemT>*
#else
inline VoxelsOnCartesianGrid<elemT>*
#endif
 get_empty_discretised_density() const;

//! Like get_empty_discretised_density, but returning a pointer to a VoxelsOnCartesianGrid
VoxelsOnCartesianGrid<elemT>* get_empty_voxels_on_cartesian_grid() const;

//TODO covariant return types
virtual DiscretisedDensity<3, elemT>* clone() const;

//! Extract a single plane
PixelsOnCartesianGrid<elemT> get_plane(const int z) const;

//! Set a single plane
void set_plane(const PixelsOnCartesianGrid<elemT>& plane, const int z);

//! is the same as get_grid_spacing(), but now returns CartesianCoordinate3D for convenience
inline CartesianCoordinate3D<float> get_voxel_size() const;

//! is the same as set_grid_spacing()
inline void set_voxel_size(const BasicCoordinate<3,float>&);

//! Growing of outer dimension only
void grow_z_range(const int min_z, const int max_z);

};


END_NAMESPACE_TOMO

#include "VoxelsOnCartesianGrid.inl"
#endif












