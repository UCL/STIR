//
// $Id$
//
/*
Copyright (C) 2005- $Date$, Hammersmith Imanet Ltd
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
  
	$Date$
	$Revision$
*/

#include "local/stir/BSplinesDetail.inl"
START_NAMESPACE_STIR

namespace BSpline {
	/////////////////////////////////////////////////////////////////////////
	
	template <int num_dimensions, typename out_elemT, typename in_elemT>
		BSplinesRegularGrid<num_dimensions,out_elemT,in_elemT>::
		BSplinesRegularGrid()
	{}
	
	
	template <int num_dimensions, typename out_elemT, typename in_elemT>
		BSplinesRegularGrid<num_dimensions,out_elemT,in_elemT>::
		~BSplinesRegularGrid()
	{}
	
	template <int num_dimensions, typename out_elemT, typename in_elemT>
		void BSplinesRegularGrid<num_dimensions, out_elemT, in_elemT>::
		set_private_values(const BasicCoordinate<num_dimensions, BSplineType> & this_type)
	{
		_spline_types = this_type;
		for ( int i = 1 ; i<=num_dimensions; ++i)
			detail::set_BSpline_values(_z1s[i],_z2s[i],_lambdas[i],this_type[i]);
	}
	
	template <int num_dimensions, typename out_elemT, typename in_elemT>
		void BSplinesRegularGrid<num_dimensions, out_elemT, in_elemT>::
		set_private_values(const BSplineType & this_type)
	{		
		for ( int i = 1 ; i<=num_dimensions; ++i)
		{
			_spline_types[i] = this_type;
			detail::set_BSpline_values(_z1s[i],_z2s[i],_lambdas[i],this_type);
		}
	}
	
	template <int num_dimensions, typename out_elemT, typename in_elemT>
		void BSplinesRegularGrid<num_dimensions, out_elemT, in_elemT> ::
		set_coef(const Array<num_dimensions,in_elemT> & input)
	{	
		detail::set_coef(_coeffs, input, _z1s, _z2s, _lambdas);
	}
	
	
	template <int num_dimensions, typename out_elemT, typename in_elemT>
		const out_elemT 
		BSplinesRegularGrid<num_dimensions, out_elemT, in_elemT>::
		operator() (const BasicCoordinate<num_dimensions,pos_type>& relative_positions) const
	{
		return detail::compute_BSplines_value(_coeffs, relative_positions, _spline_types);
	}
	
} // end of namespace BSpline

END_NAMESPACE_STIR

