//
// $Id$: $Date$
//
/*!

  \file
  \ingroup test
  
  \brief Test programme for VoxelsOnCartesianGrid and image hierarchy
    
   \author Sanida Mustafovic
   \author Kris Thielemans
   \author PARAPET project
      
   \date $Date$        
   \version $Revision$
*/

#include "VoxelsOnCartesianGrid.h"
#include "ProjDataInfo.h"
#include "ProjDataInfoCylindricalArcCorr.h"
#include "Scanner.h"
#include "IndexRange.h"
#include "tomo/round.h"

#include <iostream>
#include "RunTests.h"

START_NAMESPACE_TOMO

/*!
  \ingroup test
  \brief Test class for VoxelsOnCartesianGrid and image hierarchy

*/
class VoxelsOnCartesianGridTests : public RunTests
{
public:
  void run_tests();
};


void
VoxelsOnCartesianGridTests::run_tests()

{ 
  cerr << "Tests for VoxelsOnCartesianGrid and the image hierarchy\n";
  
  CartesianCoordinate3D<float> origin (0,1,2);  
  CartesianCoordinate3D<float> grid_spacing (3,4,5); 
  
  IndexRange<3> 
    range(CartesianCoordinate3D<int>(0,-15,-14),
          CartesianCoordinate3D<int>(4,14,15));
  
  Array<3,float> test1(range);
  
  {
    cerr << "Tests with default constructor\n";
    
    VoxelsOnCartesianGrid<float>  ob1;
    
    // Check set.* & constructor
    
    ob1.set_origin(origin);
    ob1.set_grid_spacing (grid_spacing);
    
    check_if_equal( ob1.get_grid_spacing(), grid_spacing,"test on grid_spacing");
    check_if_equal( ob1.get_origin(), origin, "test on origin");
  }
  
  {
    cerr << "Tests with 2nd constructor (array, origin, grid_spacing)\n";
    
    VoxelsOnCartesianGrid<float>  ob2(test1,origin, grid_spacing);
    test1[1][12][5] = float(5.5);
    test1[4][5][-5] = float(4.5);
    
    check_if_equal( ob2.get_grid_spacing(),grid_spacing, "test on grid_spacing");
    check_if_equal( ob2.get_origin(), origin, "test on origin");
    check_if_equal( test1.sum(), 10.F, "test on arrays");
  }
  {
    
    cerr << "Tests with 3rd constructor(index_range, origin, grid_spacing)\n";
    VoxelsOnCartesianGrid<float>  ob3(range,origin, grid_spacing);
    
    check( ob3.get_index_range() == range, "test on range");
    check_if_equal( ob3.get_grid_spacing(),grid_spacing, "test on grid_spacing");
    check_if_equal( ob3.get_origin(), origin, "test on origin");
    
  }
  
  shared_ptr<Scanner> scanner_ptr = new Scanner(Scanner::E953);
  shared_ptr<ProjDataInfo> proj_data_info_ptr = 
    ProjDataInfo::ProjDataInfoCTI(scanner_ptr, 
				  /*span=*/1, 
				  /*max_delta=*/5,
				  /*num_views=*/8,
				  /*num_tang_poss=*/16);
  
  {
    cerr << "Tests with 4th constructor with ProjDataInfo\n";
    
    const float zoom=2.3F;
    const bool make_xy_size_odd = false;
    
    VoxelsOnCartesianGrid<float>
      ob4(*proj_data_info_ptr,zoom,origin,make_xy_size_odd);
    
    IndexRange<3> obtained_range = ob4.get_index_range();
    CartesianCoordinate3D<int> low_bound, high_bound;
    check(obtained_range.get_regular_range(low_bound, high_bound), "test regular range");
    
    // KT 11/09/2001 adapted as this constructor now takes zoom into account
    const bool is_arccorrected =
      dynamic_cast<ProjDataInfoCylindricalArcCorr const *>(proj_data_info_ptr.get()) != 0;
    check(is_arccorrected, "ProjDataInfoCTI should have returned arc-corrected data");
    if (is_arccorrected)
    {
      const int radius_int = 
	round(proj_data_info_ptr->get_num_tangential_poss() * zoom/2.F);
      check_if_equal(low_bound, CartesianCoordinate3D<int>(0,-radius_int,-radius_int),
		     "test on index range: lower bounds");
      check_if_equal(high_bound, CartesianCoordinate3D<int>(30,+radius_int,+radius_int),
		     "test on index range: higher bounds");
    }
    check_if_equal(ob4.get_grid_spacing(), 
                   CartesianCoordinate3D<float>(scanner_ptr->get_ring_spacing()/2,
                                                scanner_ptr->get_default_bin_size()/zoom,
                                                scanner_ptr->get_default_bin_size()/zoom),
                   "test on grid spacing");
    check_if_equal(ob4.get_origin(), origin);
  }
  {
    
    cerr << "Tests with 5th constructor with ProjDataInfo\n";
    
    const int xy_size = 100;
    const float zoom=3.1F;
    const int min_xy = -(xy_size/2);
    const int max_xy = -(xy_size/2)+xy_size-1;
    
    VoxelsOnCartesianGrid<float>
      ob5(*proj_data_info_ptr,zoom,origin,xy_size);
    IndexRange<3> obtained_range = ob5.get_index_range();
    CartesianCoordinate3D<int> low_bound, high_bound;
    check(obtained_range.get_regular_range(low_bound, high_bound), "test regular range");
    
    check_if_equal(low_bound, CartesianCoordinate3D<int>(0,min_xy,min_xy),"test on index range: lower bounds");
    check_if_equal(high_bound, CartesianCoordinate3D<int>(30,max_xy,max_xy),"test on index range: higher bounds");
    check_if_equal(ob5.get_grid_spacing(), 
                   CartesianCoordinate3D<float>(scanner_ptr->get_ring_spacing()/2,
                                                scanner_ptr->get_default_bin_size()/zoom,
                                                scanner_ptr->get_default_bin_size()/zoom),
                   "test on grid spacing");
    check_if_equal(ob5.get_origin(), origin);
    
    {
      cerr << "Tests get_empty_voxels_on_cartesian_grid\n";
      
      shared_ptr< VoxelsOnCartesianGrid<float> > emp =ob5.get_empty_voxels_on_cartesian_grid();
      
      IndexRange<3> obtained_range2 = emp->get_index_range();
      check_if_equal( emp->get_origin(), ob5.get_origin(), "test on origin");  
      check_if_equal( emp->get_grid_spacing(), ob5.get_grid_spacing(),"test on grid_spacing");
      check(emp->get_index_range() == ob5.get_index_range(),"test on index range");
      
    }
    
    {
      cerr << "Tests get_empty_discretised_density()\n";
      
      shared_ptr<DiscretisedDensity<3,float> > emp = ob5.get_empty_discretised_density(); 
      
      VoxelsOnCartesianGrid<float>* emp1 =
        dynamic_cast<VoxelsOnCartesianGrid<float>* >(emp.get());
      check(emp1 != 0, "test on pointer conversion from get_empty_discretised_density");
      
      IndexRange<3> obtained_range3 = emp1->get_index_range();
      check_if_equal( emp->get_origin(), ob5.get_origin(), "test on origin");  
      check_if_equal( emp1->get_grid_spacing(), ob5.get_grid_spacing(),"test on grid_spacing");
      check(emp->get_index_range() == ob5.get_index_range(),"test on index range");
    }
  }
}

END_NAMESPACE_TOMO


USING_NAMESPACE_TOMO


int main()
{
  VoxelsOnCartesianGridTests tests;
  tests.run_tests();
  return tests.main_return_value();
}
