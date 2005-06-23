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
#ifndef __stir_numerics_BSplinesRegularGrid__H__
#define __stir_numerics_BSplinesRegularGrid__H__
/*!
\file 
\ingroup numerics_buildblock
\brief Implementation of the n-dimensional B-Splines Interpolation 

  \author Charalampos Tsoumpas
  \author Kris Thielemans
  
	$Date$
	$Revision$
*/

#include "stir/Array.h"
#include "stir/BasicCoordinate.h"

START_NAMESPACE_STIR

namespace BSpline {
		
	typedef double pos_type;
	
	template <int num_dimensions, typename out_elemT, typename in_elemT>
		class BSplinesRegularGrid
	{
		
private:
	
	BasicCoordinate<num_dimensions,double> _z1s;
	BasicCoordinate<num_dimensions,double> _z2s;
	BasicCoordinate<num_dimensions,double> _lambdas;
	Array<num_dimensions,out_elemT> _coeffs;  
    BasicCoordinate<num_dimensions,BSplineType> _spline_types;
	
    inline
		void
		set_coef(const Array<num_dimensions,in_elemT> & input);
		
public:
	// only there for tests
	Array<num_dimensions,out_elemT> get_coefficients() const
	{ return this->_coeffs;  }
	
	
	//! default constructor: no input
	inline BSplinesRegularGrid();
	
	//! constructor given an array as an input which estimates the Coefficients 
	inline 
		BSplinesRegularGrid(const Array<num_dimensions,in_elemT> & input,
		const BSplineType & this_type = cubic)
	{
		set_private_values(this_type);	  
		set_coef(input);		
	} 	
	
	inline 
		BSplinesRegularGrid(const Array<num_dimensions,in_elemT> & input,
		const BasicCoordinate<num_dimensions, BSplineType> & this_type)	
	{	 
		set_private_values(this_type);	  
		set_coef(input);
	}  	
	
	inline void 
		set_private_values(const BasicCoordinate<num_dimensions, BSplineType> & this_type);
	inline void 
		set_private_values(const BSplineType & this_type);
	//! destructor
	inline ~BSplinesRegularGrid();
	
	inline
		const out_elemT 
		operator() (const BasicCoordinate<num_dimensions,pos_type>& relative_positions) const;		
	};
	
} // end BSpline namespace

END_NAMESPACE_STIR

#include "local/stir/BSplinesRegularGrid.inl"

#endif
