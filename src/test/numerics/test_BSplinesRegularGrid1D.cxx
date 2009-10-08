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
  \ingroup test
  \brief tests the BSplinesRegularGrid class for the Array 1D case

  \author Charalampos Tsoumpas
  \author Kris Thielemans
  
  $Date$
  $Revision$
*/  
#include "stir/RunTests.h"
#include "stir/Array.h"
#include "stir/make_array.h"
#include "stir/IndexRange2D.h"
#include "stir/stream.h"
#include <fstream>
#include "local/stir/BSplines.h"
#include "local/stir/BSplinesRegularGrid.h"
#include <vector>
#include <algorithm>
#include <string>
#include <iostream>
#include "stir/shared_ptr.h"
#include <iomanip>
#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::ifstream;
using std::istream;
using std::setw;
#endif
START_NAMESPACE_STIR
namespace BSpline {
  /*!
    \ingroup test
    \brief A simple class to test the BSplines_coef function.
  */
  class BSplinesRegularGrid_Tests : public RunTests
  {
  public:
    BSplinesRegularGrid_Tests() 
    {}
    void run_tests();
  private:  
    template <class elemT>
    bool check_at_sample_points(const Array<1,elemT>& v,
				BSplinesRegularGrid<1, elemT, elemT>& interpolator,
				const char * const message)
    {			
      IndexRange<1> out_range(v.size());
      Array<1,elemT> out(out_range);			
      BasicCoordinate<1, elemT> relative_positions;
      for (int i=0, imax=v.size(); i<imax ; ++i)	
	{
	  relative_positions[1]=i;
	  out[i]=interpolator(relative_positions);
	}			
      cout << "IN: " << v << "OUT: " << out;    		
      return 
	check_if_equal(v, out,  message);
    }
		
    template <class elemT>
    bool check_coefficients(const Array<1,elemT>& v,
			    BSplinesRegularGrid<1, elemT, elemT>& interpolator,
			    const char * const message)
    {			
      IndexRange<1> out_range(v.size());
      Array<1,elemT> out(out_range);		
      out=interpolator.get_coefficients();			
      cout << "IN: " << v << "Coefficients: " << out;    		
      return 
	check_if_equal(v, out,  message);
    }		
  };

  void BSplinesRegularGrid_Tests::run_tests()
  {    
    cerr << "\nTesting BSplinesRegularGrid class..." << endl;
    set_tolerance(0.001);
    typedef double elemT;  		
    Array<1,elemT> const_input_sample =  make_1d_array(1., 1., 1., 1., 1., 1.);
    Array<1,elemT> linear_input_sample =  make_1d_array(1., 2., 3., 4., 5., 6.);
    Array<1,elemT> random_input_sample =  make_1d_array(-14., 8., -1., 13., -1., -2., 11., 1., -8.);		
    {
      cerr << "\nTesting BSplinesRegularGrid: Nearest Neighbour values and constructor using an 1D array as input..." << endl;	  	  	  	  
      {		 
	BSplinesRegularGrid<1, elemT, elemT> BSplinesRegularGridTest_const(
									   const_input_sample, near_n);		  
	BSplinesRegularGrid<1, elemT, elemT> BSplinesRegularGridTest_linear(
									    linear_input_sample, near_n);		  
	BSplinesRegularGrid<1, elemT, elemT> BSplinesRegularGridTest_random(
									    random_input_sample, near_n);		  
	BasicCoordinate<1,elemT> relative_positions;
				
	check_coefficients(const_input_sample, BSplinesRegularGridTest_const,
			   "check BSplines implementation for nearest interpolation");
				
	check_at_sample_points(const_input_sample, BSplinesRegularGridTest_const,
			       "check BSplines implementation for nearest interpolation");
	cerr << "At half points: "  ;
	for (elemT i=0, imax=const_input_sample.size(); i<imax ; ++i)	
	  {					
	    relative_positions[1]=i-0.5;
	    cerr <<  BSplinesRegularGridTest_const(relative_positions) <<  " " ;
	  }
	cerr << "\n\n" ;
	check_at_sample_points(linear_input_sample, BSplinesRegularGridTest_linear,
			       "check BSplines implementation for nearest interpolation");
	cerr << "At half points: "  ;
	for (elemT i=0, imax=linear_input_sample.size(); i<imax ; ++i)	
	  {					
	    relative_positions[1]=i-0.5;
	    cerr <<  BSplinesRegularGridTest_linear(relative_positions) << " " ;
	  }		
	cerr << "\n\n" ;
	check_at_sample_points(random_input_sample, BSplinesRegularGridTest_random,
			       "check BSplines implementation for nearest interpolation");
      }
    }		
    {
      cerr << "\nTesting BSplinesRegularGrid: Linear interpolation values and constructor using an 1D array as input..." << endl;	  	  	  	  
      {		 
	BSplinesRegularGrid<1, elemT, elemT> BSplinesRegularGridTest_const(
									   const_input_sample, linear);		  
	BSplinesRegularGrid<1, elemT, elemT> BSplinesRegularGridTest_linear(
									    linear_input_sample, linear);		  
	BSplinesRegularGrid<1, elemT, elemT> BSplinesRegularGridTest_random(
									    random_input_sample, linear);		  
	BasicCoordinate<1,elemT> relative_positions;
				
	check_coefficients(const_input_sample, BSplinesRegularGridTest_const,
			   "check BSplines implementation for linear interpolation");
				
	check_at_sample_points(const_input_sample, BSplinesRegularGridTest_const,
			       "check BSplines implementation for linear interpolation");
	cerr << "At half points: "  ;
	for (elemT i=0, imax=const_input_sample.size(); i<imax ; ++i)	
	  {					
	    relative_positions[1]=i-0.5;
	    cerr <<  BSplinesRegularGridTest_const(relative_positions) <<  " " ;
	  }
	cerr << "\n\n" ;
	check_at_sample_points(linear_input_sample, BSplinesRegularGridTest_linear,
			       "check BSplines implementation for linear interpolation");
	cerr << "At half points: "  ;
	for (elemT i=0, imax=linear_input_sample.size(); i<imax ; ++i)	
	  {					
	    relative_positions[1]=i-0.5;
	    cerr <<  BSplinesRegularGridTest_linear(relative_positions) << " " ;
	  }
	cerr << "\n\n" ;
	check_at_sample_points(random_input_sample, BSplinesRegularGridTest_random,
			       "check BSplines implementation for linear interpolation");
      }
    }
    {
      cerr << "\nTesting BSplinesRegularGrid: Quadratic interpolation values and constructor using an 1D array as input..." << endl;	  	  	  	  
      {		 
	BSplinesRegularGrid<1, elemT, elemT> BSplinesRegularGridTest_const(
									   const_input_sample, quadratic);		  
	BSplinesRegularGrid<1, elemT, elemT> BSplinesRegularGridTest_linear(
									    linear_input_sample, quadratic);		  
	BSplinesRegularGrid<1, elemT, elemT> BSplinesRegularGridTest_random(
									    random_input_sample, quadratic);		  
	BasicCoordinate<1,elemT> relative_positions;
				
	check_coefficients(const_input_sample, BSplinesRegularGridTest_const,
			   "check BSplines implementation for quadratic interpolation");
				
	check_at_sample_points(const_input_sample, BSplinesRegularGridTest_const,
			       "check BSplines implementation for quadratic interpolation");
	cerr << "At half points: "  ;
	for (elemT i=0, imax=const_input_sample.size(); i<imax ; ++i)	
	  {					
	    relative_positions[1]=i-0.5;
	    cerr <<  BSplinesRegularGridTest_const(relative_positions) <<  " " ;
	  }
	cerr << "\n\n" ;
	check_at_sample_points(linear_input_sample, BSplinesRegularGridTest_linear,
			       "check BSplines implementation for quadratic interpolation");
	cerr << "At half points: "  ;
	for (elemT i=0, imax=linear_input_sample.size(); i<imax ; ++i)	
	  {					
	    relative_positions[1]=i-0.5;
	    cerr <<  BSplinesRegularGridTest_linear(relative_positions) << " " ;
	  }
	cerr << "\n\n" ;
	check_at_sample_points(random_input_sample, BSplinesRegularGridTest_random,
			       "check BSplines implementation for quadratic interpolation");
      }
    }
    {
      cerr << "\nTesting BSplinesRegularGrid: Cubic interpolation values and constructor using an 1D array as input..." << endl;	  	  	  	  
      {		 
	BSplinesRegularGrid<1, elemT, elemT> BSplinesRegularGridTest_const(
									   const_input_sample, cubic);		  
	BSplinesRegularGrid<1, elemT, elemT> BSplinesRegularGridTest_linear(
									    linear_input_sample, cubic);		  
	BSplinesRegularGrid<1, elemT, elemT> BSplinesRegularGridTest_random(
									    random_input_sample, cubic);		  
	BasicCoordinate<1,elemT> relative_positions;
				
	check_coefficients(const_input_sample, BSplinesRegularGridTest_const,
			   "check BSplines implementation for cubic interpolation");
				
	check_at_sample_points(const_input_sample, BSplinesRegularGridTest_const,
			       "check BSplines implementation for cubic interpolation");
	cerr << "At half points: "  ;
	for (elemT i=0, imax=const_input_sample.size(); i<imax ; ++i)	
	  {					
	    relative_positions[1]=i-0.5;
	    cerr <<  BSplinesRegularGridTest_const(relative_positions) <<  " " ;
	  }
	cerr << "\n\n" ;
	check_at_sample_points(linear_input_sample, BSplinesRegularGridTest_linear,
			       "check BSplines implementation for cubic interpolation");
	cerr << "At half points: "  ;
	for (elemT i=0, imax=linear_input_sample.size(); i<imax ; ++i)	
	  {					
	    relative_positions[1]=i-0.5;
	    cerr <<  BSplinesRegularGridTest_linear(relative_positions) << " " ;
	  }
	cerr << "\n\n" ;
	check_at_sample_points(random_input_sample, BSplinesRegularGridTest_random,
			       "check BSplines implementation for cubic interpolation");
      }
    }
    {
      cerr << "\nTesting BSplinesRegularGrid: oMoms interpolation values and constructor using an 1D array as input..." << endl;	  	  	  	  
      {		 
	BSplinesRegularGrid<1, elemT, elemT> BSplinesRegularGridTest_const(
									   const_input_sample, oMoms);		  
	BSplinesRegularGrid<1, elemT, elemT> BSplinesRegularGridTest_linear(
									    linear_input_sample, oMoms);		  
	BSplinesRegularGrid<1, elemT, elemT> BSplinesRegularGridTest_random(
									    random_input_sample, oMoms);		  
	BasicCoordinate<1,elemT> relative_positions;
				
	check_coefficients(const_input_sample, BSplinesRegularGridTest_const,
			   "check BSplines implementation for oMoms interpolation");
				
	check_at_sample_points(const_input_sample, BSplinesRegularGridTest_const,
			       "check BSplines implementation for oMoms interpolation");
	cerr << "At half points: "  ;
	for (elemT i=0, imax=const_input_sample.size(); i<imax ; ++i)	
	  {					
	    relative_positions[1]=i-0.5;
	    cerr <<  BSplinesRegularGridTest_const(relative_positions) <<  " " ;
	  }
	cerr << "\n\n" ;
	check_at_sample_points(linear_input_sample, BSplinesRegularGridTest_linear,
			       "check BSplines implementation for oMoms interpolation");
	cerr << "At half points: "  ;
	for (elemT i=0, imax=linear_input_sample.size(); i<imax ; ++i)	
	  {					
	    relative_positions[1]=i-0.5;
	    cerr <<  BSplinesRegularGridTest_linear(relative_positions) << " " ;
	  }
	cerr << "\n\n" ;
	check_at_sample_points(random_input_sample, BSplinesRegularGridTest_random,
			       "check BSplines implementation for oMoms interpolation");
      }
    }
  }
} // end namespace BSpline

END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int main(int argc, char **argv)
{
	if (argc != 1)
	{
		cerr << "Usage : " << argv[0] << " \n";
		return EXIT_FAILURE;
	}
	BSpline::BSplinesRegularGrid_Tests tests;
	tests.run_tests();
	return tests.main_return_value();
}
