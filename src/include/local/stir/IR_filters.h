/*!
  \file
  \ingroup Filters
  \brief Implementation of the IIR and FIR filters
  
  \author Charalampos Tsoumpas
  \author Kris Thielemans
  
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2004- $Date$, Hammersmith Imanet
    See STIR/LICENSE.txt for details
*/


#include "stir/shared_ptr.h"

//#include <iterator_traits.h>
#include <vector>
#include <iostream>
#include <cmath>

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
		   const bool if_initial_exists);

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
						   const bool if_initial_exists);

template <class RandIter>
float
inline
sequence_sum(RandIter input_iterator,
	   unsigned int Nmax,
	   float pole, // to be complex as well?
	   double precision 
	   );

template <class RandIter1, 
		  class RandIter2>
void
inline 
BSplines_coef(RandIter1 c_begin_iterator, 
			   RandIter1 c_end_iterator,
			   RandIter2 input_signal_begin_iterator, 
			   RandIter2 input_signal_end_iterator);

template <class elemT>
elemT 
inline
BSpline(const elemT relative_position) ;

/*
enum enum_spline_level 
{
	constant, linear, cubic, quadratic
} spline_level;
*/


template <class elemT>
inline
elemT BSplines_weight(elemT relative_position);

template <class RandIter1, 
		  class RandIter2>
void
inline 
BSplines1DRegular(RandIter1 c_begin_iterator, 
				  RandIter1 c_end_iterator,
				  RandIter2 input_signal_begin_iterator, 
				  RandIter2 input_signal_end_iterator);



//*/
END_NAMESPACE_STIR

#include "local/stir/IR_filters.inl"