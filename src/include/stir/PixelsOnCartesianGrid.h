//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2007, Hammersmith Imanet Ltd

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/
#ifndef __stir_PixelsOnCartesianGrid_H__
#define __stir_PixelsOnCartesianGrid_H__

/*!
  \file
  \ingroup densitydata
  \brief defines the stir::PixelsOnCartesianGrid class

  \author Sanida Mustafovic
  \author Kris Thielemans (with help from Alexey Zverovich)
  \author PARAPET project



*/
#include "stir/DiscretisedDensityOnCartesianGrid.h"
#include "stir/CartesianCoordinate2D.h"

START_NAMESPACE_STIR

class ProjDataInfo;

/*!
  \ingroup densitydata
  \brief This class is used to represent pixelised densities on a
  rectangular grid (2D).

*/
template <class elemT>
class PixelsOnCartesianGrid : public DiscretisedDensityOnCartesianGrid<2, elemT>
{

public:
  //! Construct an empty PixelsOnCartesianGrid
  inline PixelsOnCartesianGrid();

  //! Construct PixelsOnCartesianGrid with the given array, origin and grid spacing
  inline PixelsOnCartesianGrid(const Array<2, elemT>& v,
                               const CartesianCoordinate3D<float>& origin,
                               const BasicCoordinate<2, float>& grid_spacing);

  //! Construct PixelsOnCartesianGrid with the given range, origin and grid spacing
  inline PixelsOnCartesianGrid(const IndexRange<2>& range,
                               const CartesianCoordinate3D<float>& origin,
                               const BasicCoordinate<2, float>& grid_spacing);

#if 0
// disabled these 2 constructors. they were never implemented.
//! The next two constructors use ProjDataInfo to obtain the relevant information
/*! \see VoxelsOnCartesianGrid(const ProjDataInfo&,const float zoom,const CartesianCoordinate3D<float>&,const CartesianCoordinate3D<int>&) for more details
 */
PixelsOnCartesianGrid(const ProjDataInfo * proj_data_info_ptr,
		      const float zoom, 
		      const CartesianCoordinate3D<float>& origin,
		      const bool make_xy_size_odd = true);

PixelsOnCartesianGrid(const ProjDataInfo * proj_data_info_ptr,
		      const float zoom,
		      const CartesianCoordinate3D<float>& origin, 
		      const int xy_size);
#endif

//! Definition of the pure virtual defined in DiscretisedDensity
#ifdef STIR_NO_COVARIANT_RETURN_TYPES
  DiscretisedDensity<2, elemT>*
#else
  PixelsOnCartesianGrid<elemT>*
#endif
  get_empty_copy() const override;

  //! Like get_empty_discretised_density, but returning a pointer to a PixelsOnCartesianGrid
  PixelsOnCartesianGrid<elemT>* get_empty_pixels_on_cartesian_grid() const;

  // TODO covariant return types

#ifdef STIR_NO_COVARIANT_RETURN_TYPES
  DiscretisedDensity<2, elemT>*
#else
  PixelsOnCartesianGrid<elemT>*
#endif
  clone() const override;

  inline int get_x_size() const;

  inline int get_y_size() const;

  inline int get_min_x() const;

  inline int get_min_y() const;

  inline int get_max_x() const;

  inline int get_max_y() const;

  //! is the same as get_grid_spacing(), but now returns CartesianCoordinate2D for convenience
  inline CartesianCoordinate2D<float> get_pixel_size() const;

  //! is the same as set_grid_spacing()
  inline void set_pixel_size(const BasicCoordinate<2, float>&) const;
};

END_NAMESPACE_STIR

#include "stir/PixelsOnCartesianGrid.inl"
#endif
