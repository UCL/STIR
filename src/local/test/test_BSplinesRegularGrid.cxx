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
\brief tests the BSplinesRegularGrid class

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
#include "stir/VectorWithOffset.h"
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
	/*	template <class elemT>
			bool check_at_sample_points(const Array<1,elemT>& v,
			BSplinesRegularGrid<1, elemT, elemT>& interpolator,
			const char * const message)
		{
			vector<elemT> out;
			const BasicCoordinate<v.size(), double> relative_positions;
			for (unsigned int i=1, imax=v.size(); i<imax ; ++i)	
			{
				relative_positions[i]=static_cast<double>(i);
				out.push_back(interpolator(relative_positions[i]));
			}
			
			cout << "IN: " << v << "OUT: " << out;    		
			return 
				check_if_equal(v, out,  message);
		}*/
	};
	void BSplinesRegularGrid_Tests::run_tests()
	{    
		cerr << "Testing BSplinesRegularGrid class..." << endl;
		set_tolerance(0.001);
		typedef double elemT;   
		/*
		static std::vector<elemT>  pre_input_sample;
		//pre_input_sample.push_back(-5); 
		pre_input_sample.push_back(-14); pre_input_sample.push_back(8);  pre_input_sample.push_back(-1);
		pre_input_sample.push_back(13);  pre_input_sample.push_back(-1); pre_input_sample.push_back(-2);
		pre_input_sample.push_back(11);  pre_input_sample.push_back(1);  pre_input_sample.push_back(-8);
		pre_input_sample.push_back(6);   pre_input_sample.push_back(11); pre_input_sample.push_back(-14);
		pre_input_sample.push_back(6);   pre_input_sample.push_back(-3); pre_input_sample.push_back(10);
		pre_input_sample.push_back(1);   pre_input_sample.push_back(7);  pre_input_sample.push_back(-2);
		pre_input_sample.push_back(-5);  pre_input_sample.push_back(-9); pre_input_sample.push_back(-9);
		pre_input_sample.push_back(6);   pre_input_sample.push_back(-5); pre_input_sample.push_back(2);
		pre_input_sample.push_back(-10); pre_input_sample.push_back(6);  pre_input_sample.push_back(-3);
		pre_input_sample.push_back(11);  pre_input_sample.push_back(11); pre_input_sample.push_back(3); 
		*/  
		Array<1,elemT> const_input_sample =  make_1d_array(1., 1., 1., 1., 1., 1.);
		{
			cerr << "Testing BSplinesRegularGrid: Nearest Neighbour values and constructor using an 1D array as input..." << endl;	  	  	  	  
			{		 
				BSplinesRegularGrid<1, elemT, elemT> BSplinesRegularGridTest1(
					const_input_sample, near_n);		  
				BasicCoordinate<1,elemT> relative_positions;

				//for (unsigned int i=1, imax=6; i<imax ; ++i)	
				//{
					relative_positions[1]=1.;//static_cast<elemT>(i);
					cerr << BSplinesRegularGridTest1(relative_positions) << endl;
				//}


				
				//check_at_sample_points(const_input_sample, BSplinesRegularGridTest1,
				//	"check BSplines implementation for nearest interpolation");
					}/*
					 {
					 std::vector<elemT> linear_input;	
					 for (elemT i=0, imax=10; i<imax ; ++i)			  
					 linear_input.push_back(i);
					 BSplinesRegularGrid<elemT, elemT> 
					 BSplinesRegularGridTesti(linear_input, near_n);
					 
					   check_at_sample_points(linear_input, BSplinesRegularGridTesti,
					   "check BSplines implementation for nearest interpolation");
					   }
					   {
					   BSplinesRegularGrid<1, elemT, elemT> 
					   BSplinesRegularGridTest(pre_input_sample, near_n);
					   check_at_sample_points(pre_input_sample, BSplinesRegularGridTest,
					   "check BSplines implementation for nearest interpolation");
		}*/
		}
		/*  {
		cerr << "Testing BSplines: Linear Interpolation values and constructor using a vector as input..." << endl;	  	  	  	  
		{
		const std::vector<elemT>  const_input_sample(10,1);	  
		BSplines1DRegularGrid<elemT, elemT> BSplines1DRegularGridTest1(
		const_input_sample.begin(), const_input_sample.end(), linear);		  
		check_at_sample_points(const_input_sample, BSplines1DRegularGridTest1,
		"check BSplines implementation for linear interpolation");
		}
		{
		std::vector<elemT> linear_input;	
		for (elemT i=0, imax=10; i<imax ; ++i)			  
		linear_input.push_back(i);
		BSplines1DRegularGrid<elemT, elemT> 
		BSplines1DRegularGridTesti(linear_input, linear);
		
		  check_at_sample_points(linear_input, BSplines1DRegularGridTesti,
		  "check BSplines implementation for linear interpolation");
		  }
		  {
		  BSplines1DRegularGrid<elemT, elemT> 
		  BSplines1DRegularGridTest(pre_input_sample, linear);
		  check_at_sample_points(pre_input_sample, BSplines1DRegularGridTest,
		  "check BSplines implementation for linear interpolation");
		  }
		  }
		  {
		  cerr << "Testing BSplines: Quadratic Interpolation values and constructor using a vector as input..." << endl;	  	  	  	  
		  {
		  const std::vector<elemT>  const_input_sample(10,1);	  
		  BSplines1DRegularGrid<elemT, elemT> BSplines1DRegularGridTest1(
		  const_input_sample.begin(), const_input_sample.end(), quadratic);		  
		  check_at_sample_points(const_input_sample, BSplines1DRegularGridTest1,
		  "check BSplines implementation for quadratic interpolation");
		  }
		  {
		  std::vector<elemT> linear_input;	
		  for (elemT i=0, imax=10; i<imax ; ++i)			  
		  linear_input.push_back(i);
		  BSplines1DRegularGrid<elemT, elemT> 
		  BSplines1DRegularGridTesti(linear_input, quadratic);
		  
			check_at_sample_points(linear_input, BSplines1DRegularGridTesti,
			"check BSplines implementation for quadratic interpolation");
			}
			{
			BSplines1DRegularGrid<elemT, elemT> 
			BSplines1DRegularGridTest(pre_input_sample, quadratic);
			check_at_sample_points(pre_input_sample, BSplines1DRegularGridTest,
			"check BSplines implementation for quadratic interpolation");
			}  
			}  
			{
			cerr << "Testing BSplines: Cubic Interpolation values and constructor using a vector as input..." << endl;	  	  	  	  
			{
			const std::vector<elemT>  const_input_sample(10,1);	  
			BSplines1DRegularGrid<elemT, elemT> BSplines1DRegularGridTest1(
			const_input_sample.begin(), const_input_sample.end(), cubic);		  
			check_at_sample_points(const_input_sample, BSplines1DRegularGridTest1,
			"check BSplines implementation for cubic interpolation");
			}
			{
			std::vector<elemT> linear_input;	
			for (elemT i=0, imax=10; i<imax ; ++i)			  
			linear_input.push_back(i);
			BSplines1DRegularGrid<elemT, elemT> 
			BSplines1DRegularGridTesti(linear_input, cubic);
			
			  check_at_sample_points(linear_input, BSplines1DRegularGridTesti,
			  "check BSplines implementation for cubic interpolation");
			  }
			  {
			  BSplines1DRegularGrid<elemT, elemT> 
			  BSplines1DRegularGridTest(pre_input_sample, cubic);
			  check_at_sample_points(pre_input_sample, BSplines1DRegularGridTest,
			  "check BSplines implementation for cubic interpolation");
			  }  
			  }  
			  {
			  cerr << "Testing BSplines: o-Moms Interpolation values and constructor using a vector as input..." << endl;	  	  	  	  
			  {
			  const std::vector<elemT>  const_input_sample(10,1);	  
			  BSplines1DRegularGrid<elemT, elemT> BSplines1DRegularGridTest1(
			  const_input_sample.begin(), const_input_sample.end(), oMoms);		  
			  check_at_sample_points(const_input_sample, BSplines1DRegularGridTest1,
			  "check BSplines implementation for o-Moms interpolation");
			  }
			  {
			  std::vector<elemT> linear_input;	
			  for (elemT i=0, imax=10; i<imax ; ++i)			  
			  linear_input.push_back(i);
			  BSplines1DRegularGrid<elemT, elemT> 
			  BSplines1DRegularGridTesti(linear_input, oMoms);
			  
				check_at_sample_points(linear_input, BSplines1DRegularGridTesti,
				"check BSplines implementation for o-Moms interpolation");
				}
				{
				linear_extrapolation(pre_input_sample);
				BSplines1DRegularGrid<elemT, elemT> 
				BSplines1DRegularGridTest(pre_input_sample, oMoms);
				check_at_sample_points(pre_input_sample, BSplines1DRegularGridTest,
				"check BSplines implementation for o-Moms or linear extrapolation interpolation");
				}  
				}    
				{
				cerr << "Testing BSplines Continuity..." << endl;	  
				std::vector<elemT>  new_input_sample(12,1) , STIR_right_output_sample, 
				STIR_left_output_sample;	  
				BSplines1DRegularGrid<elemT, elemT> BSplines1DRegularGridTests(
				new_input_sample.begin(), new_input_sample.end());  
				// test if shifted copy of the B-spline functions add to 1
				for (double inc=0; inc<1; inc+=.1)
				check_if_equal(
				BSplines_weight(+inc)+
				BSplines_weight(fabs(+inc+1))+
				BSplines_weight(fabs(+inc+2))+
				BSplines_weight(fabs(+inc-1))+
				BSplines_weight(fabs(+inc-2)),
				1., "test on B-spline function");
				std::cerr << '\n';
				const elemT epsilon = 0.01;
				for (elemT i=1, imax=10; i<imax ;++i)
				{
				STIR_left_output_sample.push_back(BSplines1DRegularGridTests(i-epsilon));	  
				STIR_right_output_sample.push_back(BSplines1DRegularGridTests(i+epsilon));
				}	 
				std::vector<elemT>:: iterator cur_iter_stir_left_out= STIR_left_output_sample.begin(), 
				cur_iter_stir_right_out= STIR_right_output_sample.begin();
				for (; cur_iter_stir_left_out!=STIR_left_output_sample.end() &&
				cur_iter_stir_right_out!=STIR_right_output_sample.end();	 
				++cur_iter_stir_left_out, ++cur_iter_stir_right_out)					
				check_if_equal(*cur_iter_stir_left_out, *cur_iter_stir_right_out,
				"check BSplines implementation");    			
				}
				{
				cerr << "Testing BSplines 1st Derivative Continuity..." << endl;	  
				std::vector<elemT>  new_input_sample, STIR_right_output_sample, STIR_left_output_sample;
				
				  for (elemT i=0, imax=14; i<imax ;++i)  new_input_sample.push_back(i);	  
				  
					BSplines1DRegularGrid<elemT, elemT> BSplines1DRegularGridTests(
					new_input_sample.begin(), new_input_sample.end());
					
					  std::cerr << '\n';
					  const elemT epsilon = 0.0001;
					  for (elemT i=1, imax=13; i<imax ;++i)
					  {
					  STIR_left_output_sample.push_back(BSplines1DRegularGridTests.BSplines_1st_der(i-epsilon));	  
					  STIR_right_output_sample.push_back(BSplines1DRegularGridTests.BSplines_1st_der(i+epsilon));
					  }	 
					  std::vector<elemT>:: iterator cur_iter_stir_left_out= STIR_left_output_sample.begin(), 
					  cur_iter_stir_right_out= STIR_right_output_sample.begin();
					  for (; cur_iter_stir_left_out!=STIR_left_output_sample.end() &&
					  cur_iter_stir_right_out!=STIR_right_output_sample.end();	 
					  ++cur_iter_stir_left_out, ++cur_iter_stir_right_out)			  
					  check_if_equal(*cur_iter_stir_left_out, *cur_iter_stir_right_out,
					  "check BSplines implementation");    	
					  }
					  {
					  cerr << "Testing BSplines values giving a vector as input..." << endl;	  
					  std::vector<elemT>  input_sample(10,1), output_sample_position, STIR_output_sample;	  	  
					  
						BSplines1DRegularGrid<elemT, elemT> BSplines1DRegularGridTests(input_sample);
						for (elemT i=0, imax=23; i<imax ;++i)	  	
						output_sample_position.push_back((i+0.5)/2.4);			
						
						  STIR_output_sample=BSplines1DRegularGridTests.BSplines_output_sequence(output_sample_position);
						  
							std::vector<elemT>:: iterator cur_iter_stir_out = STIR_output_sample.begin(), 
							cur_iter_input = input_sample.begin();
							
							  for (; cur_iter_stir_out!=STIR_output_sample.end(); ++cur_iter_stir_out)	
							  check_if_equal(*cur_iter_stir_out, (elemT)1,	
							  "check BSplines implementation");  							
							  }		
							  {
							  cerr << "Testing Linear Extrapolation giving a vector as input..." << endl;	  
							  std::vector<elemT>  input_sample(9,1);
							  *input_sample.begin()=10;
							  input_sample.push_back(10);	  
							  linear_extrapolation(input_sample);
							  check_if_equal(*input_sample.begin() *(*(input_sample.end()-1)) , (elemT)361, 
							  "check BSplines implementation");    	
  }		*/
  /*  {
  cerr << "Testing interpolation results. Look at the text files!" << endl;	  
  std::vector<elemT>  exp_input_sample;	  
  std::vector<elemT> STIR_output_sample;	
  int imax =30;
  for(int i=0; i<imax; ++i)
  exp_input_sample.push_back(exp(-(i-10.)*(i-10)/20));
  pre_input_sample = exp_input_sample;
  //linear_extrapolation(pre_input_sample);
  std::vector<elemT>:: iterator cur_iter_stir_out= STIR_output_sample.begin();
  BSplines1DRegularGrid<elemT, elemT> BSplines1DRegularGridTest(
  pre_input_sample.begin(), pre_input_sample.end(), linear);
  string output_string;	 
		output_string +=  "gaussian"; //"noisy_inter" ;	 
		ofstream out(output_string.c_str()); //output file //
		
		  if(!out)
		  {    
		  cout << "Cannot open text file.\n" ; 
		  //return EXIT_FAILURE;
		  }  
		  
			
			  //			STIR_output_sample.push_back(BSplines1DRegularGridTest(i));
			  //     cout << STIR_output_sample;     
			  out << BSplines1DRegularGridTest.spline_type <<". B-Spline \n"; 
			  out << "Position " << "\t" << " Value \n" ;
			  for (elemT i=0; i<imax+2 ;i+=1)
			  { 		
			  out << i-1-imax/2 << "\t" << exp(-(i-10.)*(i-10)/20) << "\n" ; 
			  }
			  
				out.close();       
  }*/
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
