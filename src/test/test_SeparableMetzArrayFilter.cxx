
/*!
  \file 
  \ingroup test
 
  \brief tests for the stir::SeparableMetzArrayFilter class

  \author Kris Thielemans


*/
/*
    Copyright (C) 2018, University College London
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

#include "stir/Array.h"
#include "stir/SeparableMetzArrayFilter.h"
#include "stir/IndexRange2D.h"
#include "stir/IndexRange3D.h"
#include "stir/Succeeded.h"
#include "stir/RunTests.h"

#include <iostream>
#include <algorithm>
#include <boost/static_assert.hpp>

START_NAMESPACE_STIR


/*!
  \brief Tests SeparableMetzArrayFilter functionality
  \ingroup test
  \ingroup Array

  Currently only very basic tests on normalisation
*/
class SeparableMetzArrayFilterTests : public RunTests
{
public:
  void run_tests();
private:



};
void
SeparableMetzArrayFilterTests::run_tests()
{ 
  std::cerr << "\nTesting 3D\n";
  {
    set_tolerance(.001F);
    const int size1=69;
    const int size2=200;
    const int size3=130;
    Array<3,float> test(IndexRange3D(size1,size2,size3));
#if 1
    test.fill(0.F);
    test[size1/2][size2/2][size3/2] = 1.F;
#else    
    // initialise to some arbitrary values
    {
      Array<3,float>::full_iterator iter = test.begin_all();
      /*for (int i=-100; iter != test.end_all(); ++i, ++iter)
       *iter = 1;//i*i*2.F-i-100.F;*/
      test[0][0[0]]=1;
    }
#endif
    {

      VectorWithOffset<float> fwhms(1,3);
      fwhms[1]=9.F; fwhms[2]=7.4F; fwhms[3]=5.4F;
      VectorWithOffset<float> metz_powers(1,3);
      BasicCoordinate<3,float> sampling_distances = make_coordinate(3.1F, 2.F, 2.5F);
      VectorWithOffset<int> max_kernel_sizes(1,3);
      max_kernel_sizes[1]=19; max_kernel_sizes[2]=19; max_kernel_sizes[3]=29;
      
      {
        metz_powers[1]=0.F; metz_powers[2]=0.F; metz_powers[3]=0.F;
        SeparableMetzArrayFilter<3,float> filter(fwhms, metz_powers, sampling_distances, max_kernel_sizes);

        filter(test);
        check_if_equal(1.F, test.sum(), "test if Gaussian kernel is normalised to 1");
        check(test.find_min() >= -0.005F, "test if Gaussian kernel is almost non-negative");
      }
      {
        metz_powers[1]=1.F; metz_powers[2]=2.F; metz_powers[3]=1.F;
        SeparableMetzArrayFilter<3,float> filter(fwhms, metz_powers, sampling_distances, max_kernel_sizes);

        filter(test);
        double old_tol = get_tolerance();
        set_tolerance(.01);
        check_if_equal(1.F, test.sum(), "test if Metz kernel is normalised to 1");
        set_tolerance(old_tol);
      }
    }
  }  
}

END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int main()
{
  SeparableMetzArrayFilterTests tests;
  tests.run_tests();
  return tests.main_return_value();
}
