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

#include "stir/modulo.h"
START_NAMESPACE_STIR
namespace BSpline {


  template <class IterT>
#if defined(_MSC_VER) && _MSC_VER<=1300
  float
#else
  typename  std::iterator_traits<IterT>::value_type
#endif
  cplus0(const IterT input_begin_iterator,
	 const IterT input_end_iterator, 		   
	 const double z1, const double precision, const bool periodicity)
  {
#if defined(_MSC_VER) && _MSC_VER<=1300
    typedef float out_elemT;
#else
    typedef typename std::iterator_traits<IterT>::value_type out_elemT;
#endif
	
    const int input_size = input_end_iterator - input_begin_iterator;	
    //	assert(input_size>BSplines_coef_vector.size());
    out_elemT sum=*input_begin_iterator;
    for(int i=1; 
	i<(int)ceil(log(precision)/log(fabs(z1))) && i<=2*input_size-3;
	++i)
      {  
	int index = i;
	if (periodicity==0 && i >= input_size)
	  index = 2*input_size-2-i;
	sum += *(input_begin_iterator+index)*pow(z1,i) ;
      }
    //if (periodicity==1)			
		
    return	sum/(1-pow(z1,2*input_size-2));
  }   

  template <class RandIterOut, class IterT>
  void
  BSplines_coef(RandIterOut c_begin_iterator, 
		RandIterOut c_end_iterator,
		IterT input_begin_iterator, 
		IterT input_end_iterator, 
		const double z1, const double z2, const double lamda)
  {
#if defined(_MSC_VER) && _MSC_VER<=1300
    typedef float out_elemT;
    typedef float in_elemT;
    //typedef typename _Val_Type(c_begin_iterator) out_elemT;
    //typedef typename _Val_Type(c_begin_iterator) in_elemT;
#else
    typedef typename std::iterator_traits<RandIterOut>::value_type out_elemT;
    typedef typename std::iterator_traits<RandIterOut>::value_type in_elemT;
#endif
	  	
    /*		
	cplus(c_end_iterator-c_begin_iterator, 0)
	cminus(c_end_iterator-c_begin_iterator, 0),
    */
    //const int input_size = c_end_iterator-c_begin_iterator ; //-1; //!!!

    if (z1==0 && z2==0) //Linear and Nearest Neighbour coefficients are the same to the input data
      {
	IterT current_input_iterator = input_begin_iterator;
	for(RandIterOut current_iterator=c_begin_iterator;
	    current_iterator!=c_end_iterator && current_input_iterator!=input_end_iterator; 
	    ++current_iterator,++current_input_iterator)		
	  *current_iterator= (out_elemT)(*current_input_iterator);			
		
	//	copy(input_begin_iterator, input_end_iterator, c_begin_iterator);		
      }

    else
      {
	typedef std::vector<out_elemT> c_vector_type;
	c_vector_type cplus(c_end_iterator-c_begin_iterator), 
	  cminus(c_end_iterator-c_begin_iterator);
	std::vector<double>
	  input_factor_for_cminus(1, -z1), pole_for_cplus(1, -z1), pole_for_cminus(1,-z1);
	set_to_zero(cplus);
	set_to_zero(cminus);
	std::vector<double> input_factor_for_cplus(1, (double)1);
			
	*(cplus.begin())=cplus0(
				input_begin_iterator,input_end_iterator, z1,.00001,0); //k or Nmax_precision
			
	IIR_filter(cplus.begin(), cplus.end(),
		   input_begin_iterator, input_end_iterator,
		   input_factor_for_cplus.begin(), input_factor_for_cplus.end(),
		   pole_for_cplus.begin(), pole_for_cplus.end(), 1);

	*(cminus.end()-1) = (*(cplus.end()-1) + (*(cplus.end()-2))*z1)*z1/(z1*z1-1);
	IIR_filter(cminus.rbegin(), cminus.rend(),
		   cplus.rbegin(), cplus.rend(),
		   input_factor_for_cminus.begin(), input_factor_for_cminus.end(),
		   pole_for_cminus.begin(), pole_for_cminus.end(), 1);
			
	RandIterOut current_iterator=c_begin_iterator;
	typename c_vector_type::const_iterator current_cminus_iterator=cminus.begin();
	for(;
	    current_iterator!=c_end_iterator &&	current_cminus_iterator!=cminus.end(); 
	    ++current_iterator,++current_cminus_iterator)
	  {
	    *current_iterator=*current_cminus_iterator*lamda;			
	  }
      }
  }

} // end BSpline namespace

END_NAMESPACE_STIR

