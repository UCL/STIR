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
START_NAMESPACE_STIR

namespace BSpline {
/////////////////////////////////////////////////////////////////////////
  namespace detail {
	
	template <int num_dimensions,typename out_elemT, typename in_elemT>
		void inline
		set_private_values(const BasicCoordinate<num_dimensions, BSplineType> & this_type)
	{
		this->spline_types = this_type;
		for ( int i = 1 ; i<=num_dimensions; ++i)
			detail::set_BSpline_values(this->z1s[i],this->z2s[i],this->lambdas[i],this_type[i]);
	}

	template <typename out_elemT, typename in_elemT, int num_dimensions>
		void inline
		set_private_values(const BSplineType & this_type)
	{		
		for ( int i = 1 ; i<=num_dimensions; ++i)
		{
			this->spline_types[i] = this_type;
			detail::set_BSpline_values(this->z1s[i],this->z2s[i],this->lambdas[i],this_type[i]);
		}
	}
	
	template <int num_dimensions, typename T>
	inline
	T compute_BSplines_value(const Array<num_dimensions, T>& coeffs,
		                       const BasicCoordinate<num_dimensions,pos_type>& relative_positions,
							   const BasicCoordinate<num_dimensions,BSplineType>& spline_types)
	{
		T BSplines_value=0;
		const int int_pos =(int)floor(relative_positions[1]);
		for (int k=int_pos-2; k<int_pos+3; ++k)		
		{	
			const int input_size = coeffs.size();
			int index;
			if (k<0) index=-k;
			else if (k>=input_size) index=2*input_size-2-k;
			else index = k;
			assert(0<=index && index<input_size);
			BSplines_value += 
				BSplines_weights(k-relative_positions[1], spline_types[1])*
				compute_BSplines_value(coeffs[index], 
				             cut_first_dimension(relative_positions),
							 cut_first_dimension(spline_types));	
		}
	}
	
#if defined( _MSC_VER) && _MSC_VER<=1300
#define T float
#else
	template <typename T>
#endif
		inline 
		T 
		compute_BSplines_value(const Array<1, T>& coeffs,
		             const BasicCoordinate<1,pos_type>& relative_positions,
		             const BasicCoordinate<1,BSplineType>& spline_types)
	{
		T BSplines_value=0;		
		const int int_pos =(int)floor(relative_positions[1]);
		const int input_size = coeffs.size();
		for (int k=int_pos-2; k<int_pos+3; ++k)		
		{	
			int index;
			if (k<0) index=-k;
			else if (k>=input_size) index=2*input_size-2-k;
			else index = k;
			assert(0<=index && index<input_size);
			BSplines_value += 
				BSplines_weights(k-relative_positions[1], spline_types[1])*
				coeffs[index];      
		}
	}
#if defined( _MSC_VER) && _MSC_VER<=1300
#undef T
#endif	
		template <typename out_elemT, typename in_elemT, int num_dimensions>
		void
		set_coef(Array<num_dimensions, out_elemT>& coeffs, const Array<num_dimensions, in_elemT>& input,
		const BasicCoordinate<num_dimensions,double>& z1s,
		const BasicCoordinate<num_dimensions,double>& z2s,
		const BasicCoordinate<num_dimensions,double>& lambdas)
	{		
		Array<num_dimensions,out_elemT> temp ( input.get_index_range());
	
		BSplines_coef(temp.begin(),temp.end(), 
			input.begin(), input.end(), z1s[1], z2s[1], lambdas[1]);
		
		for (int i=coeffs.get_min_index(); i<=coeffs.get_max_index(); ++i)
		{
			set_coef(coeffs[i],
				temp[i], 
				cut_first_dimension(z1s), 
				cut_first_dimension(z2s), 
				cut_first_dimension(lambdas));
		}
	}
	
	// 1d specialisation
	template <typename in_elemT>
		void
		set_coef(Array<1, in_elemT>& coeffs, const Array<1, in_elemT>& input,
		const BasicCoordinate<1,double>& z1s,
		const BasicCoordinate<1,double>& z2s,
		const BasicCoordinate<1,double>& lambdas)
	{		
		BSplines_coef(coeffs.begin(), coeffs.end(), 
			input.begin(), input.end(), z1s[1], z2s[1], lambdas[1]);
	}
	
	
#if 0
	// TODO later
	// fancy stuff to avoid having to repeat above convolution formulas for different kernels
	/*
	value of convolution of an array with a separable kernel (given by f in all dimensions)
	
	  get_value would be used normally to say get_value(coeffs,index) == coeffs[index],
	  but allows to use e.g. periodic boundary conditions or extrapolation
	*/
	
	template <typename T, typename FunctionT, int num_dimensions, class IndexingMethod>
		inline 
		T 
		convolution(const Array<num_dimensions, T>& coeffs,
		IndexingMethod get_value,
		const BasicCoordinate<num_dimensions,pos_type>& relative_positions,
		FunctionT f)
	{
		T BSplines_value=0;		
		const int int_pos =(int)floor(relative_position);
		for (int k=int_pos-2; k<int_pos+3; ++k)		
		{	
			BSplines_value += 
				f(k-relative_positions[1])*
				convolution(get_value(coeffs,index), cut_first_dimension(relative_positions), f);				
		}
	}
		
	// 1d specialisation
	template <typename T, typename FunctionT>
		inline 
		T 
		convolution(const Array<1, T>& coeffs,
		const BasicCoordinate<1,pos_type>& relative_positions,
		FunctionT f)
	{
		T BSplines_value=0;		
		const int int_pos =(int)floor(relative_position);
		for (int k=int_pos-2; k<int_pos+3; ++k)		
		{	
			int index;
			if (k<0) index=-k;
			else if (k>=input_size) index=2*input_size-2-k;
			else index = k;
			assert(0<=index && index<input_size);
			BSplines_value += 
				f(k-relative_positions[1])*
				get_value(coeffs,index);			
		}
	}
	
	template <typename T, int num_dimensions>
		inline 
		T 
		BSplines_dim(const Array<num_dimensions, T>& coeffs,
		const BasicCoordinate<num_dimensions,pos_type>& relative_positions)
	{
		convolution(coeffs, relative_positions, BSplines_weight<T>);
	}
	
	template <typename T, int num_dimensions>
		inline 
		T 
		compute_BSplines_value(const Array<num_dimensions, T>& coeffs,
		const BasicCoordinate<num_dimensions,pos_type>& relative_positions)
		/*{
		convolution(coeffs, relative_positions, BSplines_weight<T>);
        }*/

#endif

  } // end of namespace detail

  	
	/////////////////////////////////////////////////////////////////////////
template <int num_dimensions, typename out_elemT, typename in_elemT>
void
BSplinesRegularGrid<num_dimensions, out_elemT, in_elemT> ::
set_coef(const Array<num_dimensions,in_elemT> & input)
{	
	detail::set_coef(this->_coeffs, input, this->_z1s, this->_z2s, this->_lambdas);
}

template <int num_dimensions, typename out_elemT, typename in_elemT>
const out_elemT 
BSplinesRegularGrid<num_dimensions, out_elemT, in_elemT>::
operator() (const BasicCoordinate<num_dimensions,pos_type>& relative_positions) const
{
	return detail::compute_BSplines_value(this->_coeffs, relative_positions, this->_spline_types);
}
	
} // end of namespace BSpline

END_NAMESPACE_STIR
