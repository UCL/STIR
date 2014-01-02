/*
  Copyright (C) 2005 - 2009-10-08, Hammersmith Imanet Ltd
  Copyright (C) 2013, University College London
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
  \ingroup numerics_buildblock
  \brief Implementation of the B-Splines Interpolation 

  \author Kris Thielemans
  \author Charalampos Tsoumpas
*/

#include "stir/numerics/BSplinesDetail.inl"
START_NAMESPACE_STIR

namespace BSpline {

		
template <int num_dimensions, typename out_elemT, typename in_elemT, typename constantsT>
BSplinesRegularGrid<num_dimensions,out_elemT,in_elemT, constantsT>::
~BSplinesRegularGrid()
{}
	
  template <int num_dimensions, typename out_elemT, typename in_elemT, typename constantsT>
  void BSplinesRegularGrid<num_dimensions, out_elemT, in_elemT, constantsT>::
  set_private_values(const BasicCoordinate<num_dimensions, BSplineType> & this_type)
  {
    this->_spline_types = this_type;
    for ( int i = 1 ; i<=num_dimensions; ++i)
      detail::set_BSpline_values(this->_z1s[i],this->_z2s[i],this->_lambdas[i],this_type[i]);
  }
	
  template <int num_dimensions, typename out_elemT, typename in_elemT, typename constantsT>
  void BSplinesRegularGrid<num_dimensions, out_elemT, in_elemT, constantsT>::
  set_private_values(const BSplineType & this_type)
  {		
    for ( int i = 1 ; i<=num_dimensions; ++i)
      {
	this->_spline_types[i] = this_type;
	detail::set_BSpline_values(this->_z1s[i],this->_z2s[i],this->_lambdas[i],this_type);
      }
  }
	
  template <int num_dimensions, typename out_elemT, typename in_elemT, typename constantsT>
  void BSplinesRegularGrid<num_dimensions, out_elemT, in_elemT, constantsT> ::
  set_coef(const Array<num_dimensions,in_elemT> & input)
  {	
    this->_coeffs = Array<num_dimensions,out_elemT>(input.get_index_range());
    detail::set_coef(this->_coeffs, input, this->_z1s, this->_z2s, this->_lambdas);
  }
	
	
  template <int num_dimensions, typename out_elemT, typename in_elemT, typename constantsT>
  const out_elemT 
  BSplinesRegularGrid<num_dimensions, out_elemT, in_elemT, constantsT>::
  operator() (const BasicCoordinate<num_dimensions,pos_type>& relative_positions) const
  {
    return detail::compute_BSplines_value<num_dimensions, num_dimensions, in_elemT>()(this->_coeffs, relative_positions, this->_spline_types);
  }

  template <int num_dimensions, typename out_elemT, typename in_elemT, typename constantsT>
  const BasicCoordinate<num_dimensions, out_elemT> 
  BSplinesRegularGrid<num_dimensions, out_elemT, in_elemT, constantsT>::
  gradient(const BasicCoordinate<num_dimensions,pos_type>& relative_positions) const
  {
    return detail::compute_BSplines_gradient<num_dimensions, num_dimensions, in_elemT>()(this->_coeffs, relative_positions, this->_spline_types);
  }
	
} // end of namespace BSpline

END_NAMESPACE_STIR

