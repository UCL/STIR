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
  \brief tests the stir::DynamicDiscretisedDensity class

  \author Charalampos Tsoumpas
  \author Kris Thielemans
  $Date$
  $Revision$
*/


#include <iostream>

#include "stir/RunTests.h"
#include "stir/IndexRange2D.h"
#include "stir/stream.h"
#include "stir/Succeeded.h"
#include <iostream>
#include "stir/DynamicDiscretisedDensity.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/DiscretisedDensity.h"
#include "stir/shared_ptr.h"
#include "stir/TimeFrameDefinitions.h"
#include "stir/Scanner.h"
#include <utility>
#include <vector>
#include <string>

#include <algorithm>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
#endif

START_NAMESPACE_STIR
 
  class DynamicDiscretisedDensityTests : public RunTests
  {
  public:
    DynamicDiscretisedDensityTests() 
    {}
    void run_tests();
    //private:
  };

 void DynamicDiscretisedDensityTests::run_tests()
{
  {
    // Simple Test of one voxel  
  cerr << "Testing DynamicDiscretisedDensity class for one voxel..." << endl;

  set_tolerance(0.000000000000001);
  const CartesianCoordinate3D< float > origin (0.F,0.F,0.F);
  BasicCoordinate<3, float > grid_spacing ;
  grid_spacing[1] = 1.F;
  grid_spacing[2] = 1.F;
  grid_spacing[3] = 1.F;
  BasicCoordinate<3,int> sizes ;
  sizes[1]=1;
  sizes[2]=1;
  sizes[3]=1;
  IndexRange<3> range(sizes);
  
  const shared_ptr<DiscretisedDensity<3,float>  > frame1_sptr = 
    new VoxelsOnCartesianGrid<float> (range, origin,  grid_spacing) ;  
  (*frame1_sptr)[0][0][0] = 1.F;

  std::vector< std::pair< double, double > > time_frame_definitions_vector(1) ;
  std::pair< double, double > time_frame_pair(1.,1.) ;
  time_frame_definitions_vector[0]=time_frame_pair;
  const TimeFrameDefinitions time_frame_definitions(time_frame_definitions_vector);
  const double scan_start_time_in_secs_since_1970 = double(1277478034); // somewhere in June 2010...

  Scanner::Type test_scanner=Scanner::E966;
  shared_ptr<Scanner> scanner_sptr = new Scanner(test_scanner);
  DynamicDiscretisedDensity dynamic_image(time_frame_definitions, scan_start_time_in_secs_since_1970, scanner_sptr); 
  dynamic_image.set_density_sptr(frame1_sptr, 1);
  check_if_equal(dynamic_image[1][0][0][0],1.F,"check DynamicDiscretisedDensity class implementation");
  }
  {
  //  Test of two frame images, read voxel  
  cerr << "Writing DynamicDiscretisedDensity class for two frames 63x128x128..." << endl;

  set_tolerance(0.001);
  const CartesianCoordinate3D< float > origin (0.F,0.F,0.F);
  BasicCoordinate<3, float > grid_spacing ;
  grid_spacing[1] = 2.425F;
  grid_spacing[2] = 2.0594F;
  grid_spacing[3] = 2.0594F;
  BasicCoordinate<3,int> min_size, max_size;
  min_size[1]=0;  min_size[2]=-64;  min_size[3]=-64;
  max_size[1]=63;  max_size[2]=63;  max_size[3]=63;
  
  IndexRange<3> range(min_size,max_size);
  const shared_ptr<DiscretisedDensity<3,float>  > frame1_2_sptr = 
    new VoxelsOnCartesianGrid<float> (range, origin,  grid_spacing) ;  
  const shared_ptr<DiscretisedDensity<3,float>  > frame2_2_sptr = 
    new VoxelsOnCartesianGrid<float> (range, origin,  grid_spacing) ;  

      for(int k=min_size[3];k<min_size[3];++k)
        for(int j=max_size[2];j<max_size[2];++j)  
          for(int i=min_size[1];i<min_size[1];++i)
            {
              (*frame1_2_sptr)[k][j][i] = 1*(i+j*5.F-k*10.F) ;
              (*frame2_2_sptr)[k][j][i] = 2*(i+j*5.F-k*10.F) ;
            }

  std::vector< std::pair< double, double > > time_frame_definitions_vector(2) ;
  std::pair< double, double > first_time_frame_pair(1.,3.) ;
  std::pair< double, double > second_time_frame_pair(3.,6.) ;

  time_frame_definitions_vector[0]=first_time_frame_pair;
  time_frame_definitions_vector[1]=second_time_frame_pair;

  const TimeFrameDefinitions time_frame_definitions(time_frame_definitions_vector);
  const double scan_start_time_in_secs_since_1970 = double(1277478034); // somewhere in June 2010...
  Scanner::Type test_scanner=Scanner::E966;
  shared_ptr<Scanner> scanner_sptr = new Scanner(test_scanner);

  DynamicDiscretisedDensity empty_dynamic_image(time_frame_definitions,
                                                scan_start_time_in_secs_since_1970,
                                                scanner_sptr,frame1_2_sptr); 

  DynamicDiscretisedDensity dynamic_image(time_frame_definitions,
                                          scan_start_time_in_secs_since_1970,
                                          scanner_sptr); 
  dynamic_image.set_density_sptr(frame1_2_sptr, 1);
  dynamic_image.set_density_sptr(frame2_2_sptr, 2);

  string string_test("STIRtmp_dyn2f.img");
  string string_empty_test("STIRtmp_dyn2f_empty.img");

#ifdef HAVE_LLN_MATRIX
  check(dynamic_image.write_to_ecat7(string_test)==Succeeded::yes,"check DynamicDiscretisedDensity::write_to_ecat7 implementation");
#endif
  check_if_equal((empty_dynamic_image.get_density(1)).find_min(),0.F,"check DynamicDiscretisedDensity constructor implementation");          
  check_if_equal((empty_dynamic_image.get_density(1)).find_max(),0.F,"check DynamicDiscretisedDensity constructor implementation");          
  check_if_equal((empty_dynamic_image.get_density(2)).find_min(),0.F,"check DynamicDiscretisedDensity constructor implementation");          
  check_if_equal((empty_dynamic_image.get_density(2)).find_max(),0.F,"check DynamicDiscretisedDensity constructor implementation");          
}
  {
  //  Test of three frame images, read voxel  
  cerr << "Testing DynamicDiscretisedDensity class for three frames..." << endl;

  set_tolerance(0.001);
  const CartesianCoordinate3D< float > origin (0.F,0.F,0.F);
  BasicCoordinate<3, float > grid_spacing ;
  grid_spacing[1] = 2.425F;
  grid_spacing[2] = 2.0594F;
  grid_spacing[3] = 2.0594F;
  BasicCoordinate<3,int> min_size, max_size;
  min_size[1]=0;  min_size[2]=-64;  min_size[3]=-64;
  max_size[1]=63;  max_size[2]=63;  max_size[3]=63;  
  IndexRange<3> range(min_size,max_size);
  const shared_ptr<DiscretisedDensity<3,float>  > frame1_3_sptr = 
    new VoxelsOnCartesianGrid<float> (range, origin,  grid_spacing) ;  
  const shared_ptr<DiscretisedDensity<3,float>  > frame2_3_sptr = 
    new VoxelsOnCartesianGrid<float> (range, origin,  grid_spacing) ;  
  const shared_ptr<DiscretisedDensity<3,float>  > frame3_3_sptr = 
    new VoxelsOnCartesianGrid<float> (range, origin,  grid_spacing) ;  

      for(int k=min_size[3];k<min_size[3];++k)
        for(int j=max_size[2];j<max_size[2];++j)  
          for(int i=min_size[1];i<min_size[1];++i)
            {
              (*frame1_3_sptr)[k][j][i] = 1*(i+j*5.F-k*10.F) ;
              (*frame2_3_sptr)[k][j][i] = 2*(i+j*5.F-k*10.F) ;
              (*frame3_3_sptr)[k][j][i] = 3*(i+j*5.F-k*10.F) ;
            }

  std::vector< std::pair< double, double > > time_frame_definitions_vector(3) ;
  std::pair< double, double > first_time_frame_pair(1.,3.) ;
  std::pair< double, double > second_time_frame_pair(3.,6.) ;
  std::pair< double, double > third_time_frame_pair(6.5,7.) ;

  time_frame_definitions_vector[0]=first_time_frame_pair;
  time_frame_definitions_vector[1]=second_time_frame_pair;
  time_frame_definitions_vector[2]=third_time_frame_pair;

  const TimeFrameDefinitions time_frame_definitions(time_frame_definitions_vector);
  const double scan_start_time_in_secs_since_1970 = double(1277478034); // somewhere in June 2010...
  Scanner::Type test_scanner=Scanner::E966;
  shared_ptr<Scanner> scanner_sptr = new Scanner(test_scanner);
  DynamicDiscretisedDensity dynamic_image(time_frame_definitions,
                                          scan_start_time_in_secs_since_1970,
                                          scanner_sptr); 
  dynamic_image.set_density_sptr(frame1_3_sptr, 1);
  dynamic_image.set_density_sptr(frame2_3_sptr, 2);
  dynamic_image.set_density_sptr(frame3_3_sptr, 3);
#ifndef HAVE_LLN_MATRIX
  warning("write_to_ecat7 not tested as LLN library not present");
#else
  string string_test("STIRtmp_dyn3f.img");//TODO: Use the path info!!!
  //  string string_test2("./local/samples/dyn_image_write_to_ecat7_test2.img");
  //  dynamic_image.write_to_ecat7(string_test);
  check(dynamic_image.write_to_ecat7(string_test)==Succeeded::yes,"check DynamicDiscretisedDensity::write_to_ecat7 implementation");
  shared_ptr< DynamicDiscretisedDensity >  dyn_image_read_test_sptr =  
    DynamicDiscretisedDensity::read_from_file(string_test);
  const DynamicDiscretisedDensity & dyn_image_read_test = *dyn_image_read_test_sptr;
  //  dyn_image_read_test.write_to_ecat7(string_test2);

  for(int k=min_size[3];k<min_size[3];++k)
    for(int j=max_size[2];j<max_size[2];++j)  
      for(int i=min_size[1];i<min_size[1];++i)
        {
          check_if_equal(dynamic_image[1][k][j][i],(*frame1_3_sptr)[k][j][i],"check DynamicDiscretisedDensity class implementation");        
          check_if_equal(dynamic_image[2][k][j][i],(*frame2_3_sptr)[k][j][i],"check DynamicDiscretisedDensity class implementation");
          check_if_equal(dynamic_image[3][k][j][i],(*frame3_3_sptr)[k][j][i],"check DynamicDiscretisedDensity class implementation");
          check_if_equal(dyn_image_read_test[1][k][j-64][i-64],(*frame1_3_sptr)[k][j][i],"check DynamicDiscretisedDensity::read_from_file implementation"); // The written image is read in respect to its center as origin!!!
          check_if_equal(dyn_image_read_test[2][k][j-64][i-64],(*frame2_3_sptr)[k][j][i],"check DynamicDiscretisedDensity::read_from_file implementation");
          check_if_equal(dyn_image_read_test[3][k][j-64][i-64],(*frame3_3_sptr)[k][j][i],"check DynamicDiscretisedDensity::read_from_file implementation");     
        }
    check_if_equal((dynamic_image.get_time_frame_definitions()).get_end_time(1),3.,"check DynamicDiscretisedDensity class implementation");
    check_if_equal((dynamic_image.get_time_frame_definitions()).get_start_time(1),1.,"check DynamicDiscretisedDensity class implementation");
    check_if_equal((dynamic_image.get_time_frame_definitions()).get_end_time(2),6.,"check DynamicDiscretisedDensity class implementation");
    check_if_equal((dynamic_image.get_time_frame_definitions()).get_start_time(2),3.,"check DynamicDiscretisedDensity class implementation");
    check_if_equal((dynamic_image.get_time_frame_definitions()).get_end_time(3),7.,"check DynamicDiscretisedDensity class implementation");
    check_if_equal((dynamic_image.get_time_frame_definitions()).get_start_time(3),6.5,"check DynamicDiscretisedDensity class implementation");
    /* To be tested when write_time_frame_definitions() will be implemented.
    check_if_equal((dyn_image_read_test.get_time_frame_definitions()).get_end_time(1),3.,"check DynamicDiscretisedDensity class implementation");
    check_if_equal((dyn_image_read_test.get_time_frame_definitions()).get_start_time(1),1.,"check DynamicDiscretisedDensity class implementation");
    check_if_equal((dyn_image_read_test.get_time_frame_definitions()).get_end_time(2),6.,"check DynamicDiscretisedDensity class implementation");
    check_if_equal((dyn_image_read_test.get_time_frame_definitions()).get_start_time(2),3.,"check DynamicDiscretisedDensity class implementation");
    check_if_equal((dyn_image_read_test.get_time_frame_definitions()).get_end_time(3),7.,"check DynamicDiscretisedDensity class implementation");
    check_if_equal((dyn_image_read_test.get_time_frame_definitions()).get_start_time(3),6.5,"check DynamicDiscretisedDensity class implementation");
    */
#endif
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
  DynamicDiscretisedDensityTests tests;
  tests.run_tests();
  return tests.main_return_value();
}
