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
  \brief tests the BSplines_1st_der_weight function

  \author Charalampos Tsoumpas
  \author Kris Thielemans

  $Date$
  $Revision$
*/

  
#include "stir/RunTests.h"
#include "stir/Array.h"
#include "stir/IndexRange2D.h"
#include "stir/stream.h"
#include "local/stir/BSplines.h"
//#include "local/stir/BSplines_coef.h"
#include <vector>
#include <algorithm>


#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::ifstream;
using std::istream;
#endif

START_NAMESPACE_STIR

/*!
  \ingroup test
  \brief A simple class to test the BSplines_coef function.
*/
class BSplines_Tests : public RunTests
{
public:
  BSplines_Tests() 
  {}
  void run_tests();
private:
  //istream& in;
};


void BSplines_Tests::run_tests()
{  
  cerr << "Testing BSplines set of functions..." << endl;
  set_tolerance(0.001); 
  typedef double elemT;   
  {
	  BSplines1DRegularGrid<elemT, elemT> BSplines1DRegularGridTests;
	  cerr << "Testing BSplines_weight function..." << endl;
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

  {  
	  cerr << "Testing BSplines_1st_der_weight function..." << endl;
	  BSplines1DRegularGrid<elemT, elemT> BSplines1DRegularGridTests;
	  std::vector<elemT> BSplines_1st_der_weight_STIR_vector, BSplines_1st_der_weight_correct_vector, 
		  BSplines_1st_der_weight_est_vector;
	  
	  BSplines_1st_der_weight_STIR_vector.push_back(BSplines_1st_der_weight(0.));

	  for(elemT i=0.3; i<=3 ;++i)
	  	  BSplines_1st_der_weight_STIR_vector.push_back(BSplines_1st_der_weight(i));
		 	  	  	
	  BSplines_1st_der_weight_correct_vector.push_back(0.); //1
	  BSplines_1st_der_weight_correct_vector.push_back(-0.465); //2
	  BSplines_1st_der_weight_correct_vector.push_back(-0.245); //3
	  BSplines_1st_der_weight_correct_vector.push_back(0.); //4
	  
	  for ( std::vector<elemT>:: iterator cur_iter_stir_out= BSplines_1st_der_weight_STIR_vector.begin()
		  , cur_iter_test= BSplines_1st_der_weight_correct_vector.begin();
	        cur_iter_stir_out!=BSplines_1st_der_weight_STIR_vector.end() &&
				cur_iter_test!=BSplines_1st_der_weight_correct_vector.end();	  
			++cur_iter_stir_out, ++cur_iter_test)
				check_if_equal(*cur_iter_stir_out, *cur_iter_test,
				"check BSplines_1st_der_weight implementation");    		  		  		  	  		  
	 }
  {  
	  cerr << "Testing BSplines 1st Derivative analytically..." << endl;

	  
	  std::vector<elemT> BSplines_1st_der_STIR_vector, 
		  BSplines_1st_der_est_vector, new_input_sample;
	  
	  for (elemT i=0, imax=15; i<imax ;++i)
	  {	
		  new_input_sample.push_back(1);	  
	  }
	  BSplines1DRegularGrid<elemT, elemT> BSplines1DRegularGridTests(new_input_sample);
	  BSplines_1st_der_STIR_vector.push_back(0);
	  BSplines_1st_der_est_vector.push_back(0);

	  const double epsilon = 1;
	  for(elemT i=1; i<=28 ;++i)
	  {
		  BSplines_1st_der_STIR_vector.push_back(BSplines1DRegularGridTests.BSpline_1st_der(i));
		//	  cerr << BSplines1DRegularGridTests.BSpline_1st_der((i+3)/30) << endl;
		  BSplines_1st_der_est_vector.push_back(
			  (BSplines1DRegularGridTests(i+epsilon) - 
			  BSplines1DRegularGridTests(i-epsilon)) /epsilon);
		//  cerr << (BSplines1DRegularGridTests.BSplines_weight(i+0.5)) << endl;// - BSplines1DRegularGridTests(i-epsilon))/epsilon << endl;
	  }	  
	  BSplines_1st_der_STIR_vector.push_back(0);
	  BSplines_1st_der_est_vector.push_back(0);
	
	  for ( std::vector<elemT>:: iterator cur_iter_stir_out= BSplines_1st_der_est_vector.begin()
		  , cur_iter_test=BSplines_1st_der_STIR_vector.begin();
	       // cur_iter_stir_out!=BSplines_1st_der_weight_est_vector.end() &&
				cur_iter_test!=BSplines_1st_der_STIR_vector.end();	  
			++cur_iter_stir_out, ++cur_iter_test)
				check_if_equal(*cur_iter_stir_out, *cur_iter_test,
				"check BSplines_1st_der_est implementation"); 
		  		 //cout << *cur_iter_test << endl ;	  		     
	 }


  {
	  cerr << "Testing BSplines values and constructor using a vector as input..." << endl;	  
	  std::vector<elemT>  pre_input_sample/*(30,1)*/;
	  
 	  pre_input_sample.push_back(-14);
	  pre_input_sample.push_back(8);
	  pre_input_sample.push_back(-1);	  
	  pre_input_sample.push_back(13);
	  pre_input_sample.push_back(-1);
	  pre_input_sample.push_back(-2);
	  pre_input_sample.push_back(11);
	  pre_input_sample.push_back(1);
	  pre_input_sample.push_back(-8);
	  pre_input_sample.push_back(6);
	  pre_input_sample.push_back(11);
	  pre_input_sample.push_back(-14);
	  pre_input_sample.push_back(6);
	  pre_input_sample.push_back(-3);
	  pre_input_sample.push_back(10);
	  pre_input_sample.push_back(1);
	  pre_input_sample.push_back(7);
	  pre_input_sample.push_back(-2);
	  pre_input_sample.push_back(-5);
	  pre_input_sample.push_back(-9);
	  pre_input_sample.push_back(-9);
	  pre_input_sample.push_back(6);	  
	  pre_input_sample.push_back(-5);
	  pre_input_sample.push_back(2);
	  pre_input_sample.push_back(-10);
	  pre_input_sample.push_back(6);
	  pre_input_sample.push_back(-3);
	  pre_input_sample.push_back(11);
	  pre_input_sample.push_back(11);
	  pre_input_sample.push_back(3);
	  

#if 0
	  typedef Array<2,float>  input_type;
	  typedef Array<1,float> out_elemT;
	  input_type input_sample(IndexRange2D(pre_input_sample.size(),1));
#else
	  typedef Array<1,float>  input_type;
	  typedef float out_elemT;
	  input_type input_sample(pre_input_sample.size());
#endif	  
	  std::vector<out_elemT> STIR_output_sample;
	  
	  std::copy(pre_input_sample.begin(), pre_input_sample.end(), input_sample.begin_all());

	  BSplines1DRegularGrid<out_elemT, elemT> BSplines1DRegularGridTests(input_sample.begin(), input_sample.end());
	    for (elemT i=0, imax=30; i<imax ;++i)
	  	  STIR_output_sample.push_back(BSplines1DRegularGridTests(i));
	  
	  std::vector<out_elemT>:: iterator cur_iter_stir_out= STIR_output_sample.begin();
	  input_type::const_iterator cur_iter_input= input_sample.begin();

	  for (int i=0; cur_iter_stir_out!=STIR_output_sample.end() &&
		  cur_iter_input!=input_sample.end();	 
	  		    ++cur_iter_stir_out, ++cur_iter_input, ++i)			  
				{
				//check_if_equal(*cur_iter_stir_out, *cur_iter_input,
				//"check BSplines implementation");    	
//	cerr << (BSplines1DRegularGridTests.BSplines_coef_vector[i]) << endl;
					 cout << *cur_iter_stir_out << endl ;
				}
	
  }  
  {
	  cerr << "Testing BSplines Continuity..." << endl;	  
	  std::vector<elemT>  new_input_sample(16,1) , STIR_right_output_sample, 
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
	  for (elemT i=1, imax=14; i<imax ;++i)
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
			// cout << *cur_iter_stir_out << endl ;
  }
  {
	  cerr << "Testing BSplines 1st Derivative Continuity..." << endl;	  
	  std::vector<elemT>  new_input_sample, STIR_right_output_sample, STIR_left_output_sample;

	  for (elemT i=0, imax=15; i<imax ;++i)
	  {	
		  new_input_sample.push_back(i);	  
	  }
	  
	  BSplines1DRegularGrid<elemT, elemT> BSplines1DRegularGridTests(
		  new_input_sample.begin(), new_input_sample.end());
	  	  
	  std::cerr << '\n';
	  const elemT epsilon = 0.0001;
	  for (elemT i=1, imax=14; i<imax ;++i)
	  {
		  /*for (double inc=0; inc<1; inc+=.1)
			  std::cerr << BSplines1DRegularGridTests.BSpline(i+inc) << ' ';	  
		  std::cerr << '\n';*/
	  	  STIR_left_output_sample.push_back(BSplines1DRegularGridTests.BSpline_1st_der(i-epsilon));	  
		  STIR_right_output_sample.push_back(BSplines1DRegularGridTests.BSpline_1st_der(i+epsilon));
	  }
	 
	  std::vector<elemT>:: iterator cur_iter_stir_left_out= STIR_left_output_sample.begin(), 
			  cur_iter_stir_right_out= STIR_right_output_sample.begin();

	  for (; cur_iter_stir_left_out!=STIR_left_output_sample.end() &&
		  cur_iter_stir_right_out!=STIR_right_output_sample.end();	 
	  		    ++cur_iter_stir_left_out, ++cur_iter_stir_right_out)			  
				
				check_if_equal(*cur_iter_stir_left_out, *cur_iter_stir_right_out,
				"check BSplines implementation");    	
			// cout << *cur_iter_stir_out << endl ;
  }
  {
	  cerr << "Testing BSplines values giving a vector as input..." << endl;	  
	  std::vector<elemT>  input_sample(10,1), output_sample_position, STIR_output_sample;	  	  
	  
	    BSplines1DRegularGrid<elemT, elemT> BSplines1DRegularGridTests(input_sample);
		for (elemT i=0, imax=23; i<imax ;++i)	  	
				output_sample_position.push_back((i+0.5)/2.4);			
					  		
		STIR_output_sample=BSplines1DRegularGridTests.BSpline_output_sequence(output_sample_position);

		std::vector<elemT>:: iterator cur_iter_stir_out = STIR_output_sample.begin(), 
			  cur_iter_input = input_sample.begin();

	  for (; cur_iter_stir_out!=STIR_output_sample.end(); 
		  //		  && cur_iter_input!=input_sample.end(); , ++cur_iter_input
		  ++cur_iter_stir_out)	
				{	
			        check_if_equal(*cur_iter_stir_out, (elemT)1, //*cur_iter_input,			
						  "check BSplines implementation");  					
					//cerr << cur_iter_stir_out - STIR_output_sample.begin() << " " ;
					//cout << *cur_iter_stir_out << endl;
				}
  }	
	
}
END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int main(int argc, char **argv)
{
  if (argc != 1)
  {
    cerr << "Usage : " << argv[0] << " \n";
    return EXIT_FAILURE;
  }
  BSplines_Tests tests;
  tests.run_tests();
  return tests.main_return_value();
}
