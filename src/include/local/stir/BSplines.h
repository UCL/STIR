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
  \brief Implementation of the (cubic) B-Splines Interpolation 

  \author Charalampos Tsoumpas
  \author Kris Thielemans

  $Date$
  $Revision$
*/

#include "stir/shared_ptr.h"
#include <vector>
#include <algorithm>
#include <iostream>
#include <cmath>
#include "local/stir/IR_filters.h"

START_NAMESPACE_STIR

namespace BSpline {

typedef double pos_type;

enum BSplineType 
	{near_n, linear, quadratic, cubic, quartic, quintic, oMoms} ;

template <typename out_elemT, typename in_elemT>
class BSplines1DRegularGrid
{
private:
	typedef typename std::vector<out_elemT>::iterator RandIterOut; 
	int input_size; // create in the constructor 
	double z1;
	double z2;
	double lamda;
	bool if_deriv;

public:
	std::vector<out_elemT> BSplines_coef_vector;  
    BSplineType spline_type;

		
/*
void
inline  
BSplines1DRegular();
				  */
 //! default constructor: no input
  inline BSplines1DRegularGrid();
  
  //! constructor given a vector as an input, estimates the Coefficients 
  inline explicit BSplines1DRegularGrid(const std::vector<in_elemT> & input_vector);
  		
  //! constructor given a begin_ and end_ iterator as input, estimates the Coefficients 
  template <class IterT>
  inline BSplines1DRegularGrid(const IterT input_begin_iterator, 
				  const IterT input_end_iterator)
{	 
	set_private_values(cubic);	  	
	set_coef(input_begin_iterator, input_end_iterator);
}
  //! constructor given a begin_ and end_ iterator as input, estimates the Coefficients 
  template <class IterT>
  inline BSplines1DRegularGrid(const IterT input_begin_iterator, 
				  const IterT input_end_iterator, const BSplineType this_type)
  {	 
	set_private_values(this_type);	  	
	set_coef(input_begin_iterator, input_end_iterator);
  }  

  inline
	  BSplines1DRegularGrid(const std::vector<in_elemT> & input_vector, const BSplineType this_type); 
  
 //! destructor
inline ~BSplines1DRegularGrid();

  // sadly,VC6.0 needs definition of template members in the class definition
  inline void
	  set_private_values(const BSplineType this_type)
{	 
	  spline_type = this_type;	
	  if_deriv = false;
	
	switch(spline_type)
	{
	case near_n:
		z1=0.;
		z2=0.;
		break;
	case linear:
		z1=0.;
		z2=0.;
		break;
	case quadratic:
		z1 = sqrt(8.)-3.;
		z2=0.;
		break;
	case cubic:
		z1 = sqrt(3.)-2.;
		z2=0.;
		break;
	case quartic:
		z1 = sqrt(664.-sqrt(438976.))+sqrt(304.)-19.;
		z2 = sqrt(664.-sqrt(438976.))-sqrt(304.)-19.;
		break;
	case quintic:
		z1 = 0.5*(sqrt(270.-sqrt(70980.))+sqrt(105.)-13.);
		z2 = 0.5*(sqrt(270.-sqrt(70980.))-sqrt(105.)-13.);
		break;
	case oMoms:
		z1 = (sqrt(105.)-13.)/8.;	
		z2 = 0.;		
		break;
	}
	lamda = (1.-z1)*(1. - (1./z1));
	if (z2!=0.)
		lamda *= (1.-z2)*(1. - (1./z2));
}

template <class IterT>
	  inline
	  void
  set_coef(IterT input_begin_iterator, IterT input_end_iterator)
  {		
	BSplines1DRegularGrid::input_size = input_end_iterator - input_begin_iterator;
	BSplines_coef_vector.resize(input_size);
	BSplines_coef(BSplines_coef_vector.begin(),BSplines_coef_vector.end(), 
			input_begin_iterator, input_end_iterator, z1, z2, lamda);				
  }

inline 
out_elemT
BSpline(const pos_type relative_position) ;

inline 
out_elemT
BSpline_1st_der(const pos_type relative_position) ;

inline
out_elemT
BSpline_product(const int index, const pos_type relative_position);

inline
const out_elemT 
operator() (const pos_type relative_position) const;

inline
out_elemT 
operator() (const pos_type relative_position);

inline
const std::vector<out_elemT> 
BSpline_output_sequence(RandIterOut output_relative_position_begin_iterator,  //relative_position might be better float
						RandIterOut output_relative_position_end_iterator);
inline
const std::vector<out_elemT> 
BSpline_output_sequence(std::vector<pos_type> output_relative_position);
};

template <class IterT>
inline 
#if defined(_MSC_VER) && _MSC_VER<=1300
  float
#else
  typename std::iterator_traits<IterT>::value_type
#endif
cplus0(const IterT input_iterator,  
		const IterT input_end_iterator,
		double pole, const double precision, const bool periodicity);

template <class RandIterOut, class IterT>
inline  
void
BSplines_coef(RandIterOut c_begin_iterator, 
 			   RandIterOut c_end_iterator,
			   IterT input_begin_iterator, 
			   IterT input_end_iterator, 
			   const double z1, const double z2, const double lamda); // to be taken from the class

template <typename pos_type>
inline 
pos_type 
BSplines_weight(const pos_type relative_position);

template <typename pos_type>
pos_type 
oMoms_weight(const pos_type relative_position);

template <typename pos_type>
inline 
pos_type 
BSplines_1st_der_weight(const pos_type relative_position) ;

template <typename pos_type>
pos_type 
BSplines_weights(const pos_type relative_position, const BSplineType spline_type) ;

template <typename in_elemT>
inline
void
linear_extrapolation(std::vector<in_elemT> &input_vector);

//*/
} // end BSpline namespace

END_NAMESPACE_STIR

#include "local/stir/BSplines.inl"
#include "local/stir/BSplines_weights.inl"
#include "local/stir/BSplines_coef.inl"
