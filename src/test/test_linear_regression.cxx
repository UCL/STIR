
//
// $Id$: $Date$
//

/*
  A simple programme to test the linear_regression function.
  There should be a command line argument with the name of a file.
  This should contain a number of test cases for the fit.
  File contents should as follows:
    1 line with some general text
    1 line with some text for the next test case
    list_of_numbers
    1 line with some text for the next test case
    list_of_numbers
    ...

    
    where list_of_numbers is the following list of numbers 
    (white space is ignored)
    number_of_points
    coordinates
    data	 
    weights
    expected_constant  expected_scale  
    expected_chi_square  
    expected_variance_of_constant  expected_variance_of_scale  
    expected_covariance_of_constant_with_scale

  Kris Thielemans, 08/12/1999
*/

#include "pet_common.h"
#include "linear_regression.h"
#include <fstream>
#include <iostream>

const double tolerance = 1E-4F;

bool check_if_equal(double a, double b, char *str = "Should be equal")
{
  if ((fabs(b)>tolerance && fabs(a/b-1) > tolerance)
      || (fabs(b)<=tolerance && fabs(a-b) > tolerance))
  {
    cerr << "Error : values are " << a << " and " << b 
         << " (" << str<< ")"  << endl;
    return false;
  }
  else
    return true;
}


int main(int argc, char **argv)
{
  if (argc != 2)
  {
    cerr << "Usage : " << argv[0] << " filename\n"
         << "See source file for the format of this file.\n"
	 << endl;
    return 1;
  }


  ifstream in(argv[1]);
  if (!in)
  {
    cerr << argv[0] 
         << ": Error opening input file " << argv[1] << "\nExiting."
	 << endl;
    return 1;
  }
  
  cerr << "Testing linear_regression function..." << endl;

  char text[200];
  in.get(text,200);
  cerr << text << endl;
  int test_case = 0;
  
  while (in)
  {
    // first get rid of EOL
    in.getline(text,200);
    // now get real text
    in.get(text,200);
    test_case ++;
    
    int size = 0;
    in >> size;
    if (size<=0)
      break;

    cerr << text << endl;
    VectorWithOffset<float> coordinates(size);
    VectorWithOffset<float> measured_data(size);
    VectorWithOffset<float> weights(size);
    for (int i=0; i<size; i++)
      in >> coordinates[i];
    for (int i=0; i<size; i++)
      in >> measured_data[i];
    for (int i=0; i<size; i++)
      in >> weights[i];
    
    double expected_scale;
    double expected_constant;
    double expected_variance_of_scale;
    double expected_variance_of_constant;
    double expected_covariance_of_constant_with_scale;
    double expected_chi_square;
    
    in >> expected_constant >> expected_scale 
      >> expected_chi_square 
      >> expected_variance_of_constant >> expected_variance_of_scale 
      >> expected_covariance_of_constant_with_scale;
    
    
    double scale=0;
    double constant=0;
    double variance_of_scale=0;
    double variance_of_constant=0;
    double covariance_of_constant_with_scale=0;
    double chi_square = 0;
    
    linear_regression(
      constant, scale,
      chi_square,
      variance_of_constant,
      variance_of_scale,
      covariance_of_constant_with_scale,
      measured_data,
      coordinates,
      weights);
    
    if (
	!check_if_equal(expected_constant, constant, 
			"for parameter constant, should be equal") || 
	!check_if_equal(expected_scale, scale, 
			"for parameter scale, should be equal") ||
	!check_if_equal(expected_chi_square, chi_square, 
			"for parameter chi_square, should be equal") ||
	!check_if_equal(expected_variance_of_constant, variance_of_constant, 
			"for parameter variance_of_constant, should be equal") ||
	!check_if_equal(expected_variance_of_scale, variance_of_scale, 
			"for parameter variance_of_scale, should be equal") ||
	!check_if_equal(expected_covariance_of_constant_with_scale, 
			covariance_of_constant_with_scale, 
			"for parameter covariance_of_constant_with_scale should be equal")
	)
      exit(1);
  }

  cerr << "Everything fine" << endl;

  return 0;
}

