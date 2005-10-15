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
  \brief tests the DynamicDiscretisedDensity class

  \author Charalampos Tsoumpas
  \author Kris Thielemans
  $Date$
  $Revision$
*/


#include <iostream>

#include "stir/RunTests.h"
#include "stir/IndexRange2D.h"
#include "stir/stream.h"
#include <fstream>
#include "local/stir/DynamicDiscretisedDensity.h"
#include "stir/DiscretisedDensityOnCartesianGrid.h"
#include "stir/DiscretisedDensity.h"
#include "stir/shared_ptr.h"
#include "stir/TimeFrameDefinitions.h"
#include "stir/Scanner.h"
#include <utility>
#include <vector>
#include <string>

#include <algorithm>
#include <iomanip>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::ifstream;
using std::istream;
using std::setw;
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
  cerr << "Testing DynamicDiscretisedDensity class..." << endl;

  set_tolerance(0.000000000000001);
  const CartesianCoordinate3D< float > origin (0.F,0.F,0.F);
  BasicCoordinate<3, float > grid_spacing ;
  grid_spacing[1] = 1.F;
  grid_spacing[2] = 1.F;
  grid_spacing[3] = 1.F;
  BasicCoordinate<3,int> sizes ;
  sizes[1]=0;
  sizes[2]=0;
  sizes[3]=0;
  IndexRange<3> range(sizes);

  DiscretisedDensityOnCartesianGrid<3,float> >  frame1_sptr(range, origin,  grid_spacing) ;
  frame1_sptr[0][0][0]=1.F;
  shared_ptr<DiscretisedDensityOnCartesianGrid<3,float>  > * (frame1_sptr(range, origin,  grid_spacing) ;////
  
  std::vector< std::pair< double, double > > time_frame_definitions_vector(1) ;
  std::pair< double, double > time_frame_pair(1.,1.) ;
  time_frame_definitions_vector[0]=time_pair;
  const TimeFrameDefinitions time_frame_definitions(time_frame_definitions_vector);
  Type test_scanner=E966;

  shared_ptr<Scanner> * scanner_sptr(test_scanner);
  DynamicDiscretisedDensity dynamic_image(time_frame_definitions,scanner_sptr); 

  dynamic_image.set_density( *frame1_sptr, 1);
  std::string dyn_image_filename("dynamic_image_test");

  check_if_equal((*frame1_sptr)[0][0][0][1],1.F,"check DynamicDiscretisedDensity class implementation");

  return EXIT_SUCCESS;
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
