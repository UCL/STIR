//
// $Id$
//
/*!
  \file 
  \ingroup numerics_buildblock
  \brief defines the BSplines1DRegularGrid class

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
  
#include "stir/RunTests.h"
#include "local/stir/BSplines.h"
#include <vector>
#include <algorithm>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::ifstream;
using std::istream;
#endif

START_NAMESPACE_STIR


//template <elemT>
class BSplines1DRegularGrid // : public 
{
public:
	typedef float elemT; 
  BSplines1DRegularGrid()
  {}
  //elemT
  float BSpline(float x);
  BSplines_coef(RandIter1 c_begin_iterator, 
			   RandIter1 c_end_iterator,
			   RandIter2 input_signal_begin_iterator, 
			   RandIter2 input_signal_end_iterator);
  elemT inline
	  BSplines_weight(const elemT abs_relative_position);



private:
	std::vector<float> BSplines_coef_vector;
  
	//istream& in;
};


void BSplines1DRegularGrid::BSpline()
{  
  
  typedef double elemT;
  std::vector<elemT> BSplines_weight_STIR_vector, BSplines_weight_correct_vector;
  
  BSplines_weight_STIR_vector.push_back(BSplines_weight(0.));

  for(elemT i=0.3; i<=3 ;++i)
	  BSplines_weight_STIR_vector.push_back(BSplines_weight(i));
     
	  BSplines_weight_correct_vector.push_back(0.666667); //1
	  BSplines_weight_correct_vector.push_back(0.590167); //2
	  BSplines_weight_correct_vector.push_back(0.0571667); //3
	  BSplines_weight_correct_vector.push_back(0.); //4
	  
	  std::vector<elemT>:: iterator cur_iter_stir_out= BSplines_weight_STIR_vector.begin()
		  , 	  cur_iter_test= BSplines_weight_correct_vector.begin()		  ;
	  for (; cur_iter_stir_out!=BSplines_weight_STIR_vector.end() &&
		  cur_iter_test!=BSplines_weight_correct_vector.end();	  
		    ++cur_iter_stir_out, ++cur_iter_test)			  
				check_if_equal(*cur_iter_stir_out, *cur_iter_test,
				"check BSplines_weight implementation");    		  		  		  
}

END_NAMESPACE_STIR

