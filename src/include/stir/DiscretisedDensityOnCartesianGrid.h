//
// $Id$
//
#ifndef __DiscretisedDensityOnCartesianGrid_H__
#define __DiscretisedDensityOnCartesianGrid_H__

/*!
  \file 
  \ingroup densitydata 
 
  \brief defines the DiscretisedDensityOnCartesianGrid class 

  \author Sanida Mustafovic 
  \author Kris Thielemans 
  \author (help from Alexey Zverovich)
  \author PARAPET project

  $Date$

  $Revision$

*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/
#include "stir/DiscretisedDensity.h"

START_NAMESPACE_STIR

/*!
  \ingroup densitydata
  \brief This abstract class is the basis for images on a Cartesian grid.

  The only new information compared to DiscretisedDensity is the grid_spacing.
  This is currently a BasicCoordinate<num_dimensions,float> object. 
  It really should be CartesianCoordinate<num_dimensions,float> object, but
  we don't have such a class (as we can't provide names for the coordinates
  in the n-dimensional case.)
*/

template<int num_dimensions, typename elemT>
class DiscretisedDensityOnCartesianGrid:public DiscretisedDensity<num_dimensions,elemT>

{
public:
  //! Construct an empty DiscretisedDensityOnCartesianGrid
  inline DiscretisedDensityOnCartesianGrid();
  
  //! Constructor given range, grid spacing and origin
  inline DiscretisedDensityOnCartesianGrid(const IndexRange<num_dimensions>& range, 
		  const CartesianCoordinate3D<float>& origin,
		  const BasicCoordinate<num_dimensions,float>& grid_spacing);
  
  //! Return the grid_spacing
  inline const BasicCoordinate<num_dimensions,float>& get_grid_spacing() const;
  
  //! Set the grid_spacing
  inline void set_grid_spacing(const BasicCoordinate<num_dimensions,float>& grid_spacing_v);
  
  
  inline int get_x_size() const;
  
  inline int get_y_size() const;
  
  inline int get_z_size() const;
  
  inline int get_min_x() const;
  
  inline int get_min_y() const;

  inline int get_min_z() const;
  
  inline int get_max_x() const;
  
  inline int get_max_y() const;
  
  inline int get_max_z() const;

private:
  BasicCoordinate<num_dimensions,float> grid_spacing;
};

END_NAMESPACE_STIR

#include "stir/DiscretisedDensityOnCartesianGrid.inl"
#endif
