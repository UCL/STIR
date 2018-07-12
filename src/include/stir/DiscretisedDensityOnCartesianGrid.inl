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
/*!
  \file 
  \ingroup densitydata 
  \brief  inline implementations for stir::DiscretisedDensityOnCartesianGrid

  \author Sanida Mustafovic 
  \author Kris Thielemans 
  \author (help from Alexey Zverovich)
  \author PARAPET project


*/
#include "stir/round.h"

START_NAMESPACE_STIR

template<int num_dimensions, typename elemT>
DiscretisedDensityOnCartesianGrid<num_dimensions, elemT>::
DiscretisedDensityOnCartesianGrid()
: DiscretisedDensity<num_dimensions, elemT>(),grid_spacing()
{
#ifndef STIR_NO_NAMESPACES
  std::fill(grid_spacing.begin(), grid_spacing.end(), 0.F);
#else
  // hopefully your compiler understands this.
  // It attempts to avoid conflicts with Array::fill
  ::fill(grid_spacing.begin(), grid_spacing.end(), 0.F);
#endif
}

template<int num_dimensions, typename elemT>
DiscretisedDensityOnCartesianGrid<num_dimensions, elemT>::
DiscretisedDensityOnCartesianGrid
(const IndexRange<num_dimensions>& range_v, 
 const CartesianCoordinate3D<float>& origin_v,
 const BasicCoordinate<num_dimensions,float>& grid_spacing_v)
  : DiscretisedDensity<num_dimensions, elemT>(range_v,origin_v),
    grid_spacing(grid_spacing_v)
{}

template<int num_dimensions, typename elemT>
DiscretisedDensityOnCartesianGrid<num_dimensions, elemT>::
DiscretisedDensityOnCartesianGrid
(const shared_ptr < ExamInfo > & exam_info_sptr,
 const IndexRange<num_dimensions>& range_v,
 const CartesianCoordinate3D<float>& origin_v,
 const BasicCoordinate<num_dimensions,float>& grid_spacing_v)
  : DiscretisedDensity<num_dimensions, elemT>(exam_info_sptr,range_v,origin_v),
    grid_spacing(grid_spacing_v)
{}

template<int num_dimensions, typename elemT>
const BasicCoordinate<num_dimensions,float>& 
DiscretisedDensityOnCartesianGrid<num_dimensions, elemT>::
get_grid_spacing() const
{ return grid_spacing; }

template<int num_dimensions, typename elemT>
void 
DiscretisedDensityOnCartesianGrid<num_dimensions, elemT>::
set_grid_spacing(const BasicCoordinate<num_dimensions,float>& grid_spacing_v)
{
  grid_spacing = grid_spacing_v;
}

template<int num_dimensions, typename elemT>
bool
DiscretisedDensityOnCartesianGrid<num_dimensions, elemT>::
actual_has_same_characteristics(DiscretisedDensity<num_dimensions, elemT> const& other_of_base_type,
				std::string& explanation) const
{
  if (!base_type::actual_has_same_characteristics(other_of_base_type, explanation))
    return false;

  DiscretisedDensityOnCartesianGrid<num_dimensions, elemT> const& other =
    dynamic_cast<DiscretisedDensityOnCartesianGrid<num_dimensions, elemT> const&>
    (other_of_base_type);
  
  // we can now check on grid_spacing
  if (norm(other.get_grid_spacing() - this->get_grid_spacing()) > 1.E-4F*norm(this->get_grid_spacing()))
    {
      char tmp[2000];
      sprintf(tmp, "Not the same grid spacing: (%g,%g,%g) and (%g,%g,%g)",
	      other.get_grid_spacing()[1],
	      num_dimensions>1?other.get_grid_spacing()[2]:0.,
	      num_dimensions>2?other.get_grid_spacing()[3]:0.,
	      this->get_grid_spacing()[1],
	      num_dimensions>1?this->get_grid_spacing()[2]:0.,
	      num_dimensions>2?this->get_grid_spacing()[3]:0.);
      return false;
    }

  return true;
}

template<int num_dimensions, typename elemT> 
CartesianCoordinate3D<float>
DiscretisedDensityOnCartesianGrid<num_dimensions, elemT>:: 
actual_get_relative_coordinates_for_indices(const BasicCoordinate<num_dimensions,float>& indices) const
{
  const BasicCoordinate<num_dimensions,float> coord =
    this->get_grid_spacing() * indices;

  return
    CartesianCoordinate3D<float>(num_dimensions>2?coord[num_dimensions-2]:0,
				 num_dimensions>1?coord[num_dimensions-1]:0,
				 coord[num_dimensions]);
}
  
template<int num_dimensions, typename elemT> 
BasicCoordinate<num_dimensions,float>  
DiscretisedDensityOnCartesianGrid<num_dimensions, elemT>:: 
actual_get_index_coordinates_for_relative_coordinates(const CartesianCoordinate3D<float>& coords) const
{
  BasicCoordinate<num_dimensions,float> float_indices;
  // make sure that float_indices[d] = coords[d3=d+inc], with
  // with inc such that if d=num_dimensions, d+inc=3, 
  // so inc=3-num_dimensions and hence d3=d+3-num_dimensions
  for (int d=1, d3=4-num_dimensions; d<= num_dimensions; ++d, ++d3)
    {
      float_indices[d] = d3>0 ? coords[d3]:0;
    }
  return float_indices / this->get_grid_spacing();
}

END_NAMESPACE_STIR					 
