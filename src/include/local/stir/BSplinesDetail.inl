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
#include "stir/round.h"

START_NAMESPACE_STIR

namespace BSpline {
  ///// implementation functions Out Of the Class ////////
namespace detail {
		
  template <int num_dimensions, typename out_elemT, typename in_elemT>
  void
  set_coef(Array<num_dimensions, out_elemT>& coeffs, 
	   const Array<num_dimensions, in_elemT>& input,
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
  template <typename out_elemT, typename in_elemT>
  void 
  set_coef(Array<1, out_elemT>& coeffs, const Array<1, in_elemT>& input,
	   const BasicCoordinate<1,double>& z1s,
	   const BasicCoordinate<1,double>& z2s,
	   const BasicCoordinate<1,double>& lambdas)
  {				
    BSplines_coef(coeffs.begin(), coeffs.end(), 
		  input.begin(), input.end(), z1s[1], z2s[1], lambdas[1]);
  }

  template <typename pos_type>
  struct BW
  {
    typedef pos_type result_type;
    pos_type operator()(const pos_type p, const BSplineType type)
    {
      return BSplines_weights(p, type);
    }
  };

  template <typename pos_type>
  struct Bder
  {
    typedef pos_type result_type;
    pos_type operator()(const pos_type p, const BSplineType type)
    {
      return BSplines_1st_der_weight(p, type);
    }
  };

  // TODO later
  /*
    get_value could be used normally to say get_value(coeffs,index) == coeffs[index],
    but would allows to use e.g. periodic boundary conditions or extrapolation
  */
  template <int num_dimensions, int num_dimensions2, typename T, typename FunctionT,  typename SplineFunctionT>
  inline 
  typename SplineFunctionT::result_type
  spline_convolution(const Array<num_dimensions, T>& coeffs,
		     const BasicCoordinate<num_dimensions2,pos_type>& relative_positions,
		     const BasicCoordinate<num_dimensions2,BSplineType>& spline_types,
		     FunctionT f,
		     SplineFunctionT g)
  {
    const int current_dimension = num_dimensions2 - num_dimensions + 1;
    typename SplineFunctionT::result_type value;
    set_to_zero(value);
    const int kernel_left=
      (spline_types[current_dimension]==BSpline::near_n || spline_types[current_dimension]==BSpline::linear) 
      ? 0 : 2;
    const int kernel_right=
      (spline_types[current_dimension]==BSpline::near_n || spline_types[current_dimension]==BSpline::linear) 
      ? 1 : 2;      
    const int int_not_only_pos =round(floor(relative_positions[current_dimension]));
    for (int k=int_not_only_pos-kernel_left; k<=int_not_only_pos+kernel_right; ++k)		
      {	
	int index;
	if (k<coeffs.get_min_index()) index=2*coeffs.get_min_index()-k;
	else if (k>coeffs.get_max_index()) index=2*coeffs.get_max_index()-k;
	else index = k;
	assert(coeffs.get_min_index()<=index && index<=coeffs.get_max_index());
	value += 
	  g(coeffs[index], relative_positions, spline_types) *
	  f(relative_positions[current_dimension]-k, spline_types[current_dimension]);
      }
    return value ;
  }

  template <int num_dimensions2, typename T, typename FunctionT>
  inline 
  T
  spline_convolution(const Array<1, T>& coeffs,
		     const BasicCoordinate<num_dimensions2,pos_type>& relative_positions,
		     const BasicCoordinate<num_dimensions2,BSplineType>& spline_types,
		     FunctionT f)
  {
    const int current_dimension = num_dimensions2;
    T value;
    set_to_zero(value);		
    const int kernel_left=
      (spline_types[current_dimension]==BSpline::near_n || spline_types[current_dimension]==BSpline::linear) 
      ? 0 : 2;
    const int kernel_right=
      (spline_types[current_dimension]==BSpline::near_n || spline_types[current_dimension]==BSpline::linear) 
      ? 1 : 2;      
    const int int_not_only_pos =round(floor(relative_positions[current_dimension]));
    for (int k=int_not_only_pos-kernel_left; k<=int_not_only_pos+kernel_right; ++k)		
      {	
	int index;
	if (k<coeffs.get_min_index()) index=2*coeffs.get_min_index()-k;
	else if (k>coeffs.get_max_index()) index=2*coeffs.get_max_index()-k;
	else index = k;
	assert(coeffs.get_min_index()<=index && index<=coeffs.get_max_index());
	value += 
	  coeffs[index] *
	  f(relative_positions[current_dimension]-k, spline_types[current_dimension]);
      }
    return value ;
  }

  template <int num_dimensions,int num_dimensions2, typename T>
  struct 
  compute_BSplines_value
  {
    typedef T result_type;
    T operator()(const Array<num_dimensions, T>& coeffs,
		 const BasicCoordinate<num_dimensions2,pos_type>& relative_positions,
		 const BasicCoordinate<num_dimensions2,BSplineType>& spline_types) const
    {
      return
	spline_convolution(coeffs, relative_positions, spline_types,
			   BW<pos_type>(),
			   compute_BSplines_value<num_dimensions-1,num_dimensions2,T>());
    }
  };

#if defined(BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION)
#define T float
  template <>
#else
  template <typename T,int num_dimensions2>
#endif
  struct 
  compute_BSplines_value<1, num_dimensions2,T>
  {
    typedef T result_type;
    T operator()(const Array<1, T>& coeffs,
		 const BasicCoordinate<num_dimensions2,pos_type>& relative_positions,
		 const BasicCoordinate<num_dimensions2,BSplineType>& spline_types) const
    {
      return
	spline_convolution(coeffs, relative_positions, spline_types,
			   BW<pos_type>()
			   );
    }
  };
#if defined(BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION)
#undef T
#endif			

  template <int num_dimensions, int num_dimensions2, typename T>
  struct 
  compute_BSplines_gradient
  {
    typedef BasicCoordinate<num_dimensions,T> result_type;

    BasicCoordinate<num_dimensions,T>
    operator()(const Array<num_dimensions, T>& coeffs,
	       const BasicCoordinate<num_dimensions2,pos_type>& relative_positions,
	       const BasicCoordinate<num_dimensions2,BSplineType>& spline_types) const
    {
      const T first_value =
	spline_convolution(coeffs, relative_positions, spline_types,
			   Bder<pos_type>(),
			   compute_BSplines_value<num_dimensions-1,num_dimensions2,T>());
      const BasicCoordinate<num_dimensions-1,T> rest_value = 
	spline_convolution(coeffs, relative_positions, spline_types,
			   BW<pos_type>(),
			   compute_BSplines_gradient<num_dimensions-1,num_dimensions2,T>());
      return join(first_value, rest_value);
    }
  };

#if defined(BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION)
#define T float
  template <>
#else
  template <int num_dimensions2,typename T>
#endif
  struct 
  compute_BSplines_gradient<1,num_dimensions2,T>
  {
    typedef BasicCoordinate<1,T> result_type;

    BasicCoordinate<1,T>
    operator()(const Array<1, T>& coeffs,
	       const BasicCoordinate<num_dimensions2,pos_type>& relative_positions,
	       const BasicCoordinate<num_dimensions2,BSplineType>& spline_types) const
    {
      BasicCoordinate<1,T> result;
      result[1] = 
	spline_convolution(coeffs, relative_positions, spline_types,
			   Bder<pos_type>()
			   );
      return result;
    }
  };
#if defined(BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION)
#undef T
#endif			



} // end of namespace detail	
} // end of namespace BSpline

END_NAMESPACE_STIR
