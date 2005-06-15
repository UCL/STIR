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
			bool check_at_sample_points(const Array<2,elemT>& v,
			BSplinesRegularGrid<2, elemT, elemT>& interpolator,
			const char * const message)
		{	
			BasicCoordinate<2,int> min, max;
			v.get_regular_range(min,max);
			IndexRange<2> out_range(min,max);
			Array<2,elemT> out(out_range);			
			BasicCoordinate<2, elemT> relative_positions;
			for (elemT j=min[1] ; j<=max[1] ; ++j)	
				for (elemT i=min[2] ; i<=max[2] ; ++i)	
				{
					relative_positions[1]=j;
					relative_positions[2]=i;
					out[j][i]=interpolator(relative_positions);
				}			
				cout << "IN: \n" << v << "OUT: \n" << out;    		
				return 
					check_if_equal(v, out,  message);
		}

		template <class elemT>
			bool check_at_half_way(const Array<2,elemT>& v,
			BSplinesRegularGrid<2, elemT, elemT>& interpolator)
		{	
			BasicCoordinate<2,int> min, max;
			v.get_regular_range(min,max);
			IndexRange<2> out_range(min,max);
			Array<2,elemT> out(out_range);			
			BasicCoordinate<2, elemT> relative_positions;
			for (elemT j=min[1] ; j<=max[1] ; ++j)	
				for (elemT i=min[2] ; i<max[2] ; ++i)	
				{
					relative_positions[1]=j+0.5;
					relative_positions[2]=i+0.5;
					out[j][i]=interpolator(relative_positions);
				}		
				cout << "Checking BSplines implementation at half way:\n" ;
				cout << "IN: \n" << v << "OUT: \n" << out;
				return true;
		}
		
		template <class elemT>
			bool check_coefficients(const Array<2,elemT>& v,
			BSplinesRegularGrid<2, elemT, elemT>& interpolator,
			const char * const message)
		{	
			const Array<2,elemT> out=interpolator.get_coefficients();			
			cout << "IN: \n" << v << "Coefficients: \n" << out;    		
			return 
				check_if_equal(v, out,  message);
		}		
	};
	void BSplinesRegularGrid_Tests::run_tests()
	{    
		cerr << "\nTesting BSplinesRegularGrid class..." << endl;
		set_tolerance(0.001);
		typedef double elemT;  		
		Array<1,elemT> const_1D  =  make_1d_array(1., 1., 1., 1., 1., 1.);
		Array<1,elemT> linear_1D =  make_1d_array(1., 2., 3., 4., 5., 6.);
		Array<1,elemT> random_1D_1 =  make_1d_array(-14., 8., -1., 13., -1., -2., 11., 1., -8.);		
		Array<1,elemT> random_1D_2 =  make_1d_array(6., 11., -14., 6., -3., 10., 1., 7., -2.);		
		Array<1,elemT> random_1D_3 =  make_1d_array(-5., -9., -9., 6., -5., 2., -10., 6., -3.);		
		Array<1,elemT> random_1D_4 =  make_1d_array(11., 8., -1., -1., 12., 11., 11., 3., 1.);		
		Array<1,elemT> random_1D_5 =  make_1d_array(8., -1., 13., -1., -2., 11., 1., -8., -14.);		
		Array<1,elemT> random_1D_6 =  make_1d_array(-14., 8., -1., 13., -1., -2., 1., -8., 11.);		
		Array<1,elemT> random_1D_7 =  make_1d_array(13., -1., -2., -14.,  11., 1., -8., 8., -1.);		

		Array<2,elemT> const_input_sample =  make_array(const_1D, 
														const_1D, 
														const_1D, 
														const_1D, 
														const_1D, 
														const_1D);
		Array<2,elemT> linear_const_input_sample =  make_array(linear_1D, 
														linear_1D, 
														linear_1D, 
														linear_1D, 
														linear_1D, 
														linear_1D);
		Array<2,elemT> random_input_sample =  make_array(random_1D_1, 
														random_1D_2, 
														random_1D_3, 
														random_1D_4, 
														random_1D_5, 
														random_1D_6,
														random_1D_7);
		
		{
			cerr << "\nTesting BSplinesRegularGrid: Nearest Neighbour values and constructor using a 2D array as input..." << endl;	  	  	  	  
			{		 
				BSplinesRegularGrid<2, elemT, elemT> BSplinesRegularGridTest_const(
					const_input_sample, near_n);		  
				BSplinesRegularGrid<2, elemT, elemT> BSplinesRegularGridTest_linear_const(
					linear_const_input_sample, near_n);		  
				BSplinesRegularGrid<2, elemT, elemT> BSplinesRegularGridTest_random(
					random_input_sample, near_n);		  
				BasicCoordinate<2,elemT> relative_positions;
				
				check_coefficients(const_input_sample, BSplinesRegularGridTest_const,
					"check BSplines implementation for nearest interpolation");				
				check_at_sample_points(const_input_sample, BSplinesRegularGridTest_const,
					"check BSplines implementation for nearest interpolation");
				check_at_half_way(const_input_sample, BSplinesRegularGridTest_const);
				check_at_sample_points(linear_const_input_sample, BSplinesRegularGridTest_linear_const,
					"check BSplines implementation for nearest interpolation");
				check_at_half_way(linear_const_input_sample, BSplinesRegularGridTest_linear_const);
				check_at_sample_points(random_input_sample, BSplinesRegularGridTest_random,
					"check BSplines implementation for nearest interpolation");				
				check_at_half_way(random_input_sample, BSplinesRegularGridTest_random);
			}			
		}
		{
			cerr << "\nTesting BSplinesRegularGrid: linear interpolation values and constructor using a 2D array as input..." << endl;	  	  	  	  
			{		 
				BSplinesRegularGrid<2, elemT, elemT> BSplinesRegularGridTest_const(
					const_input_sample, linear);		  
				BSplinesRegularGrid<2, elemT, elemT> BSplinesRegularGridTest_linear_const(
					linear_const_input_sample, linear);		  
				BSplinesRegularGrid<2, elemT, elemT> BSplinesRegularGridTest_random(
					random_input_sample, linear);		  
				BasicCoordinate<2,elemT> relative_positions;
				
				check_coefficients(const_input_sample, BSplinesRegularGridTest_const,
					"check BSplines implementation for linear interpolation");				
				check_at_sample_points(const_input_sample, BSplinesRegularGridTest_const,
					"check BSplines implementation for linear interpolation");
				check_at_half_way(const_input_sample, BSplinesRegularGridTest_const);
				check_at_sample_points(linear_const_input_sample, BSplinesRegularGridTest_linear_const,
					"check BSplines implementation for linear interpolation");
				check_at_half_way(linear_const_input_sample, BSplinesRegularGridTest_linear_const);
				check_at_sample_points(random_input_sample, BSplinesRegularGridTest_random,
					"check BSplines implementation for linear interpolation");				
				check_at_half_way(random_input_sample, BSplinesRegularGridTest_random);
			}			
		}
		{
			cerr << "\nTesting BSplinesRegularGrid: quadratic interpolation values and constructor using a 2D array as input..." << endl;	  	  	  	  
			{		 
				BSplinesRegularGrid<2, elemT, elemT> BSplinesRegularGridTest_const(
					const_input_sample, quadratic);		  
				BSplinesRegularGrid<2, elemT, elemT> BSplinesRegularGridTest_linear_const(
					linear_const_input_sample, quadratic);		  
				BSplinesRegularGrid<2, elemT, elemT> BSplinesRegularGridTest_random(
					random_input_sample, quadratic);		  
				BasicCoordinate<2,elemT> relative_positions;
				
				check_coefficients(const_input_sample, BSplinesRegularGridTest_const,
					"check BSplines implementation for quadratic interpolation");				
				check_at_sample_points(const_input_sample, BSplinesRegularGridTest_const,
					"check BSplines implementation for quadratic interpolation");
				check_at_half_way(const_input_sample, BSplinesRegularGridTest_const);
				check_at_sample_points(linear_const_input_sample, BSplinesRegularGridTest_linear_const,
					"check BSplines implementation for quadratic interpolation");
				check_at_half_way(linear_const_input_sample, BSplinesRegularGridTest_linear_const);
				check_at_sample_points(random_input_sample, BSplinesRegularGridTest_random,
					"check BSplines implementation for quadratic interpolation");				
				check_at_half_way(random_input_sample, BSplinesRegularGridTest_random);
			}			
		}
		{
			cerr << "\nTesting BSplinesRegularGrid: Cubic interpolation values and constructor using a 2D array as input..." << endl;	  	  	  	  
			{		 
				BSplinesRegularGrid<2, elemT, elemT> BSplinesRegularGridTest_const(
					const_input_sample, cubic);		  
				BSplinesRegularGrid<2, elemT, elemT> BSplinesRegularGridTest_linear_const(
					linear_const_input_sample, cubic);		  
				BSplinesRegularGrid<2, elemT, elemT> BSplinesRegularGridTest_random(
					random_input_sample, cubic);		  
				BasicCoordinate<2,elemT> relative_positions;
				
				check_coefficients(const_input_sample, BSplinesRegularGridTest_const,
					"check BSplines implementation for cubic interpolation");				
				check_at_sample_points(const_input_sample, BSplinesRegularGridTest_const,
					"check BSplines implementation for cubic interpolation");
				check_at_half_way(const_input_sample, BSplinesRegularGridTest_const);
				check_at_sample_points(linear_const_input_sample, BSplinesRegularGridTest_linear_const,
					"check BSplines implementation for cubic interpolation");
				check_at_half_way(linear_const_input_sample, BSplinesRegularGridTest_linear_const);
				check_at_sample_points(random_input_sample, BSplinesRegularGridTest_random,
					"check BSplines implementation for cubic interpolation");				
				check_at_half_way(random_input_sample, BSplinesRegularGridTest_random);
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
