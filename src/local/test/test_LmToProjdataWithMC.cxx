
//
// $Id: 
//
/*!

  \file
  \ingroup mytests

  \brief Test programme for LmToProjdataWithMC functions

  \author Sanida Mustafovic
  \author PARAPET project

  $Date: 
  $Revision$
*/

#include "stir/RunTests.h"
#include "stir/Scanner.h"
#include "stir/Bin.h"
#include "local/stir/Quaternion.h"
#include <iostream>
#include <iomanip>

#include "local/stir/listmode/LmToProjDataWithMC.h"

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::setw;
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

  const int Ring_A = 5;
  const int Ring_B = 6;
  const int det1 = 144;
  const int det2 = 60;

  CartesianCoordinate3D<float> coord_1;
  CartesianCoordinate3D<float> coord_2;
  
  // normally one cannot access private memebers of in the class buI have made it public while testing
  LmToProjDataWithMCObject.find_cartesian_coordinates_given_scanner_coordinates (coord_1,coord_2,
							Ring_A,Ring_B, 
							det1,det2, 
							*scanner);

  CartesianCoordinate3D<float> tmp = (coord_2-coord_1);
  tmp.x() = 5*tmp.x();
  tmp.y() = 5*tmp.y();
  tmp.z() = 5*tmp.z();

 CartesianCoordinate3D<float> ttt = (coord_2-coord_1);
  ttt.x() = 2*ttt.x();
  ttt.y() = 2*ttt.y();
  ttt.z() = 2*ttt.z();

  CartesianCoordinate3D<float> coord_1_new = coord_1 + tmp;
  CartesianCoordinate3D<float> coord_2_new = coord_1 + ttt;


  int det1_f, det2_f,ring1_f, ring2_f;

  LmToProjDataWithMCObject.find_scanner_coordinates_given_cartesian_coordinates(det1_f, det2_f, ring1_f, ring2_f,
							coord_1_new, coord_2_new, 
							*scanner);
  cerr<< " WARNING for this check values 1 and 2 can be swapped" <<endl;
  check_if_equal( det1_f, det1, "test on det1");
  check_if_equal( det2_f, det2, "test on det2");
  check_if_equal( Ring_A, ring1_f, "test on ring1");
  check_if_equal( Ring_B, ring2_f, "test on ring1");


}
END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int main()
{
  LmToProjDataWithMCTests tests;
  tests.run_tests();
  return tests.main_return_value();
}

