//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2009-07-08, Hammersmith Imanet Ltd
    Copyright (C) 2011-07-01 - 2011, Kris Thielemans
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
  \brief  inline implementation for stir::DiscretisedDensity

  \author Sanida Mustafovic 
  \author Kris Thielemans 
  \author (help from Alexey Zverovich)
  \author PARAPET project


*/
#include "stir/round.h"
#include <string>
#include <typeinfo>

START_NAMESPACE_STIR

template<int num_dimensions, typename elemT>
DiscretisedDensity<num_dimensions,elemT>::DiscretisedDensity()
{}

template<int num_dimensions, typename elemT>
DiscretisedDensity<num_dimensions, elemT>::
DiscretisedDensity(const IndexRange<num_dimensions>& range_v,
		   const CartesianCoordinate3D<float>& origin_v)
  : Array<num_dimensions,elemT>(range_v),
    origin(origin_v)    
{}

template<int num_dimensions, typename elemT>
DiscretisedDensity<num_dimensions, elemT>::
DiscretisedDensity(const shared_ptr < ExamInfo > & exam_info_sptr,
                   const IndexRange<num_dimensions>& range_v,
                   const CartesianCoordinate3D<float>& origin_v)
  : ExamData(exam_info_sptr),
    Array<num_dimensions,elemT>(range_v),
    origin(origin_v)
{}

template<int num_dimensions, typename elemT>
void
DiscretisedDensity<num_dimensions, elemT>::
set_origin(const CartesianCoordinate3D<float> &origin_v)
{
  origin = origin_v;
}

template<int num_dimensions, typename elemT>
const CartesianCoordinate3D<float>& 
DiscretisedDensity<num_dimensions, elemT>::
get_origin()  const 
{ return origin; }


template<int num_dimensions, typename elemT>
bool
DiscretisedDensity<num_dimensions, elemT>::
actual_has_same_characteristics(DiscretisedDensity<num_dimensions, elemT> const& other,
				std::string& explanation) const
{
  if (typeid(other) != typeid(*this))
    {
      explanation = "Different type of data";
      return false;
    }

  if (norm(other.get_origin() - this->get_origin()) > 1.E-2)
    { 
      explanation = "Not the same origin.";
      return false;
    }
  if (other.get_index_range() != this->get_index_range())
    {
      explanation = "Not the same index ranges.";
      return false;
    }
  return true;
}

template<int num_dimensions, typename elemT>
bool
DiscretisedDensity<num_dimensions, elemT>::
has_same_characteristics(DiscretisedDensity<num_dimensions, elemT> const& other,
			 std::string& explanation) const
{
  return this->actual_has_same_characteristics(other, explanation);
}

template<int num_dimensions, typename elemT>
bool
DiscretisedDensity<num_dimensions, elemT>::
has_same_characteristics(DiscretisedDensity<num_dimensions, elemT> const& other) const
{
  std::string explanation;
  return this->actual_has_same_characteristics(other, explanation);
}

template<int num_dimensions, typename elemT>
bool 
DiscretisedDensity<num_dimensions, elemT>::
operator ==(const self_type& that) const
{
  return
    this->has_same_characteristics(that) &&
    base_type::operator==(that);
}
  
template<int num_dimensions, typename elemT>
bool 
DiscretisedDensity<num_dimensions, elemT>::
operator !=(const self_type& that) const
{
  return !((*this) == that);
}

template<int num_dimensions, typename elemT>
CartesianCoordinate3D<float>
DiscretisedDensity<num_dimensions, elemT>::
get_relative_coordinates_for_indices(const BasicCoordinate<num_dimensions,int>& indices) const
{
  return
    this->
    actual_get_relative_coordinates_for_indices(BasicCoordinate<num_dimensions,float>(indices));
}

template<int num_dimensions, typename elemT>
CartesianCoordinate3D<float>
DiscretisedDensity<num_dimensions, elemT>::
get_relative_coordinates_for_indices(const BasicCoordinate<num_dimensions,float>& indices) const
{
  return
    this->
    actual_get_relative_coordinates_for_indices(indices);
}

template<int num_dimensions, typename elemT>
BasicCoordinate<num_dimensions,float>
DiscretisedDensity<num_dimensions, elemT>::
get_index_coordinates_for_physical_coordinates(const CartesianCoordinate3D<float>& coords) const
{
  return
    this->actual_get_index_coordinates_for_relative_coordinates(coords - this->get_origin());
}

template<int num_dimensions, typename elemT>
BasicCoordinate<num_dimensions,float>
DiscretisedDensity<num_dimensions, elemT>::
get_index_coordinates_for_relative_coordinates(const CartesianCoordinate3D<float>& coords) const
{
  return
    this->actual_get_index_coordinates_for_relative_coordinates(coords);
}

template<int num_dimensions, typename elemT>
BasicCoordinate<num_dimensions,int>
DiscretisedDensity<num_dimensions, elemT>::
get_indices_closest_to_relative_coordinates(const CartesianCoordinate3D<float>& coords) const
{
  return
    round(this->actual_get_index_coordinates_for_relative_coordinates(coords));
}

template<int num_dimensions, typename elemT>
CartesianCoordinate3D<float>
DiscretisedDensity<num_dimensions, elemT>::
get_physical_coordinates_for_indices(const BasicCoordinate<num_dimensions,int>& indices) const
{
  return
    this->get_relative_coordinates_for_indices(indices) + this->get_origin();
}

template<int num_dimensions, typename elemT>
CartesianCoordinate3D<float>
DiscretisedDensity<num_dimensions, elemT>::
get_physical_coordinates_for_indices(const BasicCoordinate<num_dimensions,float>& indices) const
{
  return
    this->get_relative_coordinates_for_indices(indices) + this->get_origin();
}


template<int num_dimensions, typename elemT>
BasicCoordinate<num_dimensions,int>
DiscretisedDensity<num_dimensions, elemT>::
get_indices_closest_to_physical_coordinates(const CartesianCoordinate3D<float>& coords) const
{
  return
    this->get_indices_closest_to_relative_coordinates(coords - this->get_origin());
}

template<int num_dimensions, typename elemT>
CartesianCoordinate3D<float>
DiscretisedDensity<num_dimensions, elemT>::
swap_axes_based_on_orientation(const CartesianCoordinate3D<float>& coords,
                               const PatientPosition patient_position)
{
  CartesianCoordinate3D<float> flip_coords = coords;
  // STIR coordinates run:
  // x: left to right as you face the scanner (as seen from the bed side)
  // y: top to bottom of the scanner
  // z: running from deep in the scanner out along the bed (as seen from the bed
  //    side)
  // ITK coordinates are defined w.r.t LPS
  switch (patient_position.get_position())
    {
    case PatientPosition::unknown_position: // If unknown, assume HFS
    case PatientPosition::HFS:              // HFS means currently in patient LPI
      flip_coords.z() *= -1;
      break;

    case PatientPosition::HFP:              // HFP means currently in patient RAI
      flip_coords.x() *= -1;
      flip_coords.y() *= -1;
      flip_coords.z() *= -1;
      break;

    case PatientPosition::FFS:              // FFS means currently in patient RPS
      flip_coords.x() *= -1;
      break;

    case PatientPosition::FFP:              // FFP means currently in patient LAS
      flip_coords.y() *= -1;
      break;

    default:
      throw std::runtime_error("Unsupported patient position, can't convert to LPS.");
    }
  return flip_coords;
}

template<int num_dimensions, typename elemT>
CartesianCoordinate3D<float>
DiscretisedDensity<num_dimensions, elemT>::
get_LPS_coordinates_for_physical_coordinates(const CartesianCoordinate3D<float>& coords) const
{
  return swap_axes_based_on_orientation
    (coords, this->get_exam_info().patient_position);
}

template<int num_dimensions, typename elemT>
CartesianCoordinate3D<float>
DiscretisedDensity<num_dimensions, elemT>::
get_LPS_coordinates_for_indices(const BasicCoordinate<num_dimensions, float>& indices) const
{
  return this->get_LPS_coordinates_for_physical_coordinates
    (this->get_physical_coordinates_for_indices(indices));
}

template<int num_dimensions, typename elemT>
CartesianCoordinate3D<float>
DiscretisedDensity<num_dimensions, elemT>::
get_LPS_coordinates_for_indices(const BasicCoordinate<num_dimensions, int>& indices) const
{
  return get_LPS_coordinates_for_indices(BasicCoordinate<num_dimensions, float>(indices));
}

template<int num_dimensions, typename elemT>
CartesianCoordinate3D<float>
DiscretisedDensity<num_dimensions, elemT>::
get_physical_coordinates_for_LPS_coordinates(const CartesianCoordinate3D<float>& coords) const
{
  // operation is symmetric
  return this->get_LPS_coordinates_for_physical_coordinates(coords);
}

template<int num_dimensions, typename elemT>
BasicCoordinate<num_dimensions,float>
DiscretisedDensity<num_dimensions, elemT>::
get_index_coordinates_for_LPS_coordinates(const CartesianCoordinate3D<float>& coords) const
{
  return this->get_index_coordinates_for_physical_coordinates
    (this->get_physical_coordinates_for_LPS_coordinates(coords));
}

template<int num_dimensions, typename elemT>
BasicCoordinate<num_dimensions, int>
DiscretisedDensity<num_dimensions, elemT>::
get_indices_closest_to_LPS_coordinates(const CartesianCoordinate3D<float>& coords) const
{
  return round(this->get_index_coordinates_for_LPS_coordinates(coords));
}

END_NAMESPACE_STIR
