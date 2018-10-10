//
//
/*!
  \file
  \ingroup test

  \brief Test program for LmToProjdataWithMC functions
  \author Sanida Mustafovic
  \author Kris Thielemans

*/
/*
    Copyright (C) 2003- 2003, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/
#include "stir/RunTests.h"
#include "stir_experimental/Quaternion.h"
#include "stir_experimental/listmode/LmToProjDataWithMC.h"

#include <iostream>
#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
#endif


START_NAMESPACE_STIR

class LmToProjDataWithMCTests: public RunTests
{
public:  
  void run_tests();
};




void
LmToProjDataWithMCTests::run_tests()
{
  const char * const par_filename = "test.par";
  LmToProjDataWithMC LmToProjDataWithMCObject(par_filename);
  shared_ptr<Scanner> scanner = new Scanner(Scanner::E966);
  for ( int Ring_A = 1; Ring_A < 10; Ring_A++)
    for ( int Ring_B = 1; Ring_B < 10; Ring_B++)
      for ( int det1 =1; det1 <=10; det1 ++)
	for ( int det2 =100; det2 <=110; det2 ++)
	  
	{
	  CartesianCoordinate3D<float> coord_1;
	  CartesianCoordinate3D<float> coord_2;
	  
	  // normally one cannot access private members of in the class but I have made it public while testing
	  LmToProjDataWithMCObject.find_cartesian_coordinates_given_scanner_coordinates (coord_1,coord_2,
	    Ring_A,Ring_B, 
	    det1,det2, 
	    *scanner);
	  
	  const CartesianCoordinate3D<float> coord_1_new = coord_1 + (coord_2-coord_1)*5;
	  const CartesianCoordinate3D<float> coord_2_new = coord_1 + (coord_2-coord_1)*2;
	  
	  int det1_f, det2_f,ring1_f, ring2_f;
	  
	  LmToProjDataWithMCObject.find_scanner_coordinates_given_cartesian_coordinates(det1_f, det2_f, ring1_f, ring2_f,
	    coord_1_new, coord_2_new, 
	    *scanner);
	  if (det1_f == det1 && Ring_A == ring1_f)
	  { 
	    check_if_equal( det1_f, det1, "test on det1");
	    check_if_equal( Ring_A, ring1_f, "test on ring1");
	    check_if_equal( det2_f, det2, "test on det2");
	    check_if_equal( Ring_B, ring2_f, "test on ring1");
	  }
	  else
	  {
	    check_if_equal( det2_f, det1, "test on det1");
	    check_if_equal( Ring_B, ring1_f, "test on ring1");
	    check_if_equal( det1_f, det2, "test on det2");
	    check_if_equal( Ring_A, ring2_f, "test on ring1");
	  }
	}
}
END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int main()
{
  LmToProjDataWithMCTests tests;
  tests.run_tests();
  return tests.main_return_value();
}

