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

#include "stir/shared_ptr.h"
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

template <class elemT>
inline
elemT BSplines_weight(elemT relative_position);

template <class elemT>
elemT 
inline
BSplines_1st_der_weight(const elemT abs_relative_position) ;

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

#include "local/stir/BSplines.inl"