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
#include <list>

#include <cmath>

START_NAMESPACE_STIR

template <class BiIterType1,
		  class BiIterType2,
	      class BiIterType3,
		  class BiIterType4>
void 
inline 
IIR_filter(BiIterType1 output_begin_iterator, 
		   BiIterType1 output_end_iterator,
		   const BiIterType2 input_begin_iterator, 
		   const BiIterType2 input_end_iterator,
		   const BiIterType3 input_factor_begin_iterator,
		   const BiIterType3 input_factor_end_iterator,
		   const BiIterType4 pole_begin_iterator,
		   const BiIterType4 pole_end_iterator,
		   const bool if_initial_exists);

template <class BiIterType1,
		  class BiIterType2,
	      class BiIterType3>		 
void 
inline 
FIR_filter(BiIterType1 output_begin_iterator, 
						   BiIterType1 output_end_iterator,
						   const BiIterType2 input_begin_iterator, 
						   const BiIterType2 input_end_iterator,
						   const BiIterType3 input_factor_begin_iterator,
						   const BiIterType3 input_factor_end_iterator,
						   const bool if_initial_exists);

template <class BiIterType>
float
inline
sequence_sum(BiIterType input_iterator,
	   unsigned int Nmax,
	   float pole, // to be complex as well?
	   double precision 
	   );

template <class BiIterType1, 
		  class BiIterType2>
void
inline 
Bsplines_coef(BiIterType1 c_begin_iterator, 
			   BiIterType1 c_end_iterator,
			   BiIterType2 input_signal_begin_iterator, 
			   BiIterType2 input_signal_end_iterator);

//*/
END_NAMESPACE_STIR

#include "local/stir/IR_filters.inl"