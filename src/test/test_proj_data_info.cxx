//
// $Id$: $Date$
//
/*!

  \file
  \ingroup test

  \brief Test programme for ProjDataInfoCylindricalArcCorr

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author PARAPET project

  \date $Date$
  \version $Revision$
*/

#include "ProjDataInfoCylindricalArcCorr.h"
#include "RunTests.h"
#include "Scanner.h"
#include "Bin.h"
#include <iostream>

#ifndef TOMO_NO_NAMESPACES
using std::cerr;
#endif

START_NAMESPACE_TOMO

/*!
  \ingroup test
  \brief Test class for ProjDataInfoCylindricalArcCorr
*/

class ProjDataInfoCylindricalArcCorrTests: public RunTests
{
public:  
  void run_tests();
};


void
ProjDataInfoCylindricalArcCorrTests::run_tests()

{ 
  cerr << "Testing ProjDataInfoCylindricalArcCorr\n";
  {
    // Test on the empty constructor
    
    ProjDataInfoCylindricalArcCorr ob1;
    
    // Test on set.* & get.* + constructor
    const float test_tangential_sampling = 1.5;
    //const float test_azimuthal_angle_sampling = 10.1;
    
    ob1.set_tangential_sampling(test_tangential_sampling);
    // Set_azimuthal_angle_sampling
    // ob1.set_azimuthal_angle_sampling(test_azimuthal_angle_sampling);
    
    
    check_if_equal( ob1.get_tangential_sampling(), test_tangential_sampling,"test on tangential_sampling");
    //check_if_zero( ob1.get_azimuthal_angle_sampling() - test_azimuthal_angle_sampling, " test on azimuthal_angle_sampling");
    
  }
  {
    shared_ptr<Scanner> scanner_ptr = new Scanner(Scanner::E953);
    
    VectorWithOffset<int> num_axial_pos_per_segment(-1,1);
    VectorWithOffset<int> min_ring_diff(-1,1); 
    VectorWithOffset<int> max_ring_diff(-1,1);
    // simulate span=3 for segment 0, span=1 for segment 2
    num_axial_pos_per_segment[-1]=14;
    num_axial_pos_per_segment[0]=31;
    num_axial_pos_per_segment[1]=14;
    min_ring_diff[-1] = min_ring_diff[-1] = -2;
    min_ring_diff[ 0] = -1; max_ring_diff[ 0] = 1;
    min_ring_diff[+1] = max_ring_diff[+1] = +2;
    const int num_views = 96;
    const int num_tangential_poss = 128;
    
    const float bin_size = 1.2F;
    
    
    //Test on the constructor
    ProjDataInfoCylindricalArcCorr
      ob2(scanner_ptr, bin_size, 
      num_axial_pos_per_segment, min_ring_diff, max_ring_diff,
      num_views,num_tangential_poss);
    
    check_if_equal( ob2.get_tangential_sampling(), bin_size,"test on tangential_sampling");
    check_if_equal( ob2.get_azimuthal_angle_sampling() , _PI/num_views, " test on azimuthal_angle_sampling");
    check_if_equal( ob2.get_axial_sampling(1),  scanner_ptr->get_ring_spacing(), "test on axial_sampling");
    check_if_equal( ob2.get_axial_sampling(0), scanner_ptr->get_ring_spacing()/2, "test on axial_sampling for segment0");
    
    {
      // KT 25/10/2000 use bin
      // segment 0
      Bin bin(0,10,10,20);
      float theta = ob2.get_tantheta(bin);
      float phi = ob2.get_phi(bin); 
      // Get t
      float t = ob2.get_t(bin);
      //! Get s
      float s = ob2.get_s(bin);
      
      check_if_equal( theta, 0.F,"test on get_tantheta, seg 0");
      check_if_equal( phi, 10*ob2.get_azimuthal_angle_sampling(), " get_phi , seg 0");
      // KT 25/10/2000 adjust to new convention
      const float ax_pos_origin =
	(ob2.get_min_axial_pos_num(0) + ob2.get_max_axial_pos_num(0))/2.F;
      check_if_equal( t, (10-ax_pos_origin)*ob2.get_axial_sampling(0) , "get_t, seg 0");
      check_if_equal( s, 20*ob2.get_tangential_sampling() , "get_s, seg 0");
    }
    {
      // KT 25/10/2000 use bin      
      // Segment 1
      Bin bin (1,10,10,20);
      float theta = ob2.get_tantheta(bin);
      float phi = ob2.get_phi(bin); 
      // Get t
      float t = ob2.get_t(bin);
      // Get s
      float s = ob2.get_s(bin);
      
      float thetatest = 2*ob2.get_axial_sampling(1)/(2*sqrt(square(scanner_ptr->get_ring_radius())-square(s)));
      
      check_if_equal( theta, thetatest,"test on get_tantheta, seg 1");
      check_if_equal( phi, 10*ob2.get_azimuthal_angle_sampling(), " get_phi , seg 1");
      // KT 25/10/2000 adjust to new convention
      const float ax_pos_origin =
	(ob2.get_min_axial_pos_num(1) + ob2.get_max_axial_pos_num(1))/2.F;
      check_if_equal( t, (10-ax_pos_origin)/sqrt(1+square(thetatest))*ob2.get_axial_sampling(1) , "get_t, seg 1");
      check_if_equal( s, 20*ob2.get_tangential_sampling() , "get_s, seg 1");
    }
    
  }
}

END_NAMESPACE_TOMO


USING_NAMESPACE_TOMO

int main()
{
  ProjDataInfoCylindricalArcCorrTests tests;
  tests.run_tests();
  return tests.main_return_value();
}
