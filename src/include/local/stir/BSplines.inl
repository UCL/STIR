//
// $Id$
//
/*!
  \file 
  \ingroup numerics_buildblock
  \brief Implementation of the (cubic) B-Splines Interpolation 

  \author Charalampos Tsoumpas
  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
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

using namespace std;

START_NAMESPACE_STIR

template <class RandIter1,
		  class RandIter2,
	      class RandIter3,
		  class RandIter4>
void 
inline 
IIR_filter(RandIter1 output_begin_iterator, 
		   RandIter1 output_end_iterator,
		   const RandIter2 input_begin_iterator, 
		   const RandIter2 input_end_iterator,
		   const RandIter3 input_factor_begin_iterator,
		   const RandIter3 input_factor_end_iterator,
		   const RandIter4 pole_begin_iterator,
		   const RandIter4 pole_end_iterator,
		   const bool if_initial_exists)
{
	// The input should be initialised to 0
//	if(output_begin_iterator==output_end_iterator)
//		warning("No output signal is given./n");

	if(if_initial_exists==false) 
	*output_begin_iterator=*input_factor_begin_iterator*(*input_begin_iterator);

	RandIter1 current_output_iterator = output_begin_iterator ;
	RandIter2 current_input_iterator = input_begin_iterator ;
	
	for(++current_output_iterator, ++current_input_iterator; 
	     current_output_iterator != output_end_iterator &&
		 current_input_iterator != input_end_iterator;
		 ++current_output_iterator,	++current_input_iterator)
		{
			RandIter2 current_current_input_iterator = current_input_iterator;
			for(RandIter3 current_input_factor_iterator = input_factor_begin_iterator;				
				current_input_factor_iterator != input_factor_end_iterator;
				++current_input_factor_iterator,--current_current_input_iterator
				)
				{					
					(*current_output_iterator) += (*current_input_factor_iterator)*
												   (*current_current_input_iterator);
				if (current_current_input_iterator==input_begin_iterator)
					break;
				}

			RandIter4 current_pole_iterator = pole_begin_iterator;
			RandIter1 current_feedback_iterator = current_output_iterator ;
			
			for(--current_feedback_iterator ;
				current_pole_iterator != pole_end_iterator 			
					;++current_pole_iterator,--current_feedback_iterator)
				{							
					(*current_output_iterator) -= (*current_pole_iterator)*
													(*current_feedback_iterator);			
					if(current_feedback_iterator==output_begin_iterator)
					break;
				}					
		}
}

template <class RandIter1,
		  class RandIter2,
	      class RandIter3>		 
void 
inline 
FIR_filter(RandIter1 output_begin_iterator, 
		   RandIter1 output_end_iterator,
		   const RandIter2 input_begin_iterator, 
		   const RandIter2 input_end_iterator,
		   const RandIter3 input_factor_begin_iterator,
		   const RandIter3 input_factor_end_iterator,
		   const bool if_initial_exists)
{
	IIR_filter(output_begin_iterator, output_end_iterator, 
		   input_begin_iterator, input_end_iterator, 
		   input_factor_begin_iterator, input_factor_end_iterator,
		   output_begin_iterator, output_begin_iterator,if_initial_exists);
}
template <class RandIter>
float
inline
sequence_sum(RandIter input_iterator,
	   unsigned int Nmax,
	   float pole, // to be complex as well?
	   double precision 
	   )
{
	const int Nmax_precision = 4;//(int)ceil(log(precision)/log(abs(pole)));
	float sum=*input_iterator;
	for(int i=1; 
			i!=Nmax && i<=Nmax_precision //&& current_input_iterator!=end 
				; 							 ++i)
	  	  sum += *(++input_iterator)*pow(pole,i) ;
	return sum;
}   

template <class RandIter1, 
		  class RandIter2>
void
inline //temporarily
BSplines_coef(RandIter1 c_begin_iterator, 
			   RandIter1 c_end_iterator,
			   RandIter2 input_signal_begin_iterator, 
			   RandIter2 input_signal_end_iterator)
{

#ifdef _MSC_VER<=1300

	typedef float c_value_type;
	typedef float input_value_type;
	//typedef typename _Val_Type(c_begin_iterator) c_value_type;
	//typedef typename _Val_Type(c_begin_iterator) input_value_type;
#else
	typedef typename std::iterator_traits<RandIter1>::value_type c_value_type;
	typedef typename std::iterator_traits<RandIter1>::value_type input_value_type;
#endif
	  	
	std::vector<c_value_type> 
/*		
		cplus(c_end_iterator-c_begin_iterator, 0)
		cminus(c_end_iterator-c_begin_iterator, 0),
*/
		cplus, cminus, input_factor_for_cminus, pole_for_cplus, pole_for_cminus;
	
	std::vector<input_value_type> input_factor_for_cplus;
  
	float z1 = -2. + sqrt(3.);
	pole_for_cplus.push_back(-z1);
	pole_for_cminus.push_back(-z1);
	input_factor_for_cplus.push_back(1);
	input_factor_for_cminus.push_back(-z1);
	RandIter1 cplus
	
	for(RandIter1 current_iterator=c_begin_iterator; 
	    current_iterator!=c_end_iterator; 
		++current_iterator)
		{
			cplus.push_back(0);
			cminus.push_back(0);
		}
	
	  
	const int k=2*cplus.size()-3;
    *(cplus.begin())=sequence_sum(input_signal_begin_iterator,k,z1,.0001)/(1-pow(z1,k+1)) ; //k or Nmax_precision
	
	IIR_filter(cplus.begin(), cplus.end(),
		input_signal_begin_iterator, input_signal_end_iterator,
		input_factor_for_cplus.begin(), input_factor_for_cplus.end(),
		pole_for_cplus.begin(), pole_for_cplus.end(), 1);

	*--cminus.end() = z1*(*--cplus.end() + z1*(*--(--cplus.end())))/(1-z1*z1);
	IIR_filter(cminus.rbegin(), cminus.rend(),
		cplus.rbegin(), cplus.rend(),
		input_factor_for_cminus.begin(), input_factor_for_cminus.end(),
		pole_for_cminus.begin(), pole_for_cminus.end(), 1);
	
	for(RandIter1 current_iterator=c_begin_iterator, current_cminus_iterator=cminus.begin(); 
	    current_iterator!=c_end_iterator &&	current_iterator!=cminus.end(); 
		++current_iterator,++current_cminus_iterator)
		{
			*current_iterator=*current_cminus_iterator*6;			
		}
}

template <class elemT>
elemT 
inline
BSplines_weight(const elemT abs_relative_position
					//, enum enum_spline_level
					) 
{
	if (abs_relative_position>=2)
		return 0;
	if (abs_relative_position>=1)		
		return pow((2-abs_relative_position),3)/6;
	if (abs_relative_position>=0)		
		return 2./3. + (0.5*abs_relative_position-1)*abs_relative_position*abs_relative_position;
	else
	//	warning("BSplineWeight does not take negative values as input!");
		return EXIT_FAILURE;
//	-100;
}
template <class elemT>
elemT 
inline
BSplines_1st_der_weight(const elemT abs_relative_position) 
{
	if (abs_relative_position>=2)
		return 0;
	if (abs_relative_position>=1)		
		return -0.5*(-2. + abs_relative_position)*(-2. + abs_relative_position);
	if (abs_relative_position>=0)		
		return abs_relative_position*(1.5*abs_relative_position-2.);
	else
	//	warning("BSplines_1st_der_weight does not take negative values as input!");
		return EXIT_FAILURE;
//	-100;
}


template <class elemT>
elemT 
inline
BSpline(const elemT relative_position) 
{	
	elemT BSpline_value;
	const int max_size = BSpline1DRegularGrid::BSpline_coefficients.size();
/*	assert(relative_position-max_size>0)
	if (relative_position-max_size>0)
		warning("New sampling position out of range");
		*/
	for (int k=(int)relative_position-1; k<(int)relative_position+2 && k<=max_size; ++k)		
	{
		if (k==-1) continue;
		BSpline_value += BSpline1DRegularGrid::BSplines_weight(fabs(k-position))* //fabs for double and float
		BSpline1DRegularGrid::BSpline_coefficients_list[k];
	}
	return BSpline_value;
}


/* recursively
template <class RandIter>
float
inline
sequence_sum(RandIter input_iterator,
	   unsigned int Nmax,
	   float pole // to be complex as well
	   )
{
	  static float sum=0;
	  if (Nmax=!0)
	  {
		  sum = *input_iterator*pole ;
		  sum += sequence_sum(++input_iterator,--Nmax, pole);                                                      		  
		  return sum;
	  }
	  else
		  return sum;
}   
*/

END_NAMESPACE_STIR
