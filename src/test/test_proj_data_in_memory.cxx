//
//
/*!

  \file
  \ingroup test

  \brief Test program for stir::ProjDataInMemory

  \author Kris Thielemans
  \author Daniel Deidda

*/
/*
    Copyright (C) 2015, 2020 University College London
    Copyright (C) 2020, National Physical Laboratory
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

#include "stir/ProjDataInMemory.h"
#include "stir/ExamInfo.h"
#include "stir/ProjDataInfo.h"
#include "stir/Sinogram.h"
#include "stir/Viewgram.h"
#include "stir/Succeeded.h"
#include "stir/RunTests.h"
#include "stir/Scanner.h"

START_NAMESPACE_STIR


/*!
  \ingroup test
  \brief Test class for ProjDataInMemory
*/
class ProjDataInMemoryTests: public RunTests
{
public:
  void run_tests();
};

void
ProjDataInMemoryTests::
run_tests()
{
  std::cerr << "-------- Testing ProjDataInMemory --------\n";
  shared_ptr<Scanner> scanner_sptr(new Scanner(Scanner::E953));
    
  shared_ptr<ProjDataInfo> proj_data_info_sptr
    (ProjDataInfo::ProjDataInfoCTI(scanner_sptr,
		                    /*span*/1, 10,/*views*/ 96, /*tang_pos*/128, /*arc_corrected*/ true)
     );
  shared_ptr<ExamInfo> exam_info_sptr(new ExamInfo);
  

  // construct with filling to 0
  ProjDataInMemory proj_data(exam_info_sptr, proj_data_info_sptr);
  {
    Sinogram<float> sinogram = proj_data.get_sinogram(0,0);
    check_if_equal(sinogram.find_min(),
                   0.F,
                   "test constructor and get_sinogram");
  }

  const float value = 1.2F;
  // test fill(float)
  {
    proj_data.fill(value);
    Viewgram<float> viewgram = proj_data.get_viewgram(0,0);
    check_if_equal(viewgram.find_min(),
                   value,
                   "test fill(float) and get_viewgram");
  }
  
  // test set_viewgram
  {
    Viewgram<float> viewgram = proj_data.get_empty_viewgram(1,1);
    viewgram.fill(value*2);
    check(proj_data.set_viewgram(viewgram) == Succeeded::yes,
          "test set_viewgram succeeded");

    Viewgram<float> viewgram2 = proj_data.get_viewgram(1,1);
    check_if_equal(viewgram2.find_min(),
                   viewgram.find_min(),
                   "test set/get_viewgram");
  }

  // test making a copy 
  {
    ProjDataInMemory proj_data2(proj_data);
    check_if_equal(proj_data2.get_viewgram(0,0).find_max(),
                   proj_data.get_viewgram(0,0).find_max(),
                   "test 1 for copy-constructor and get_viewgram");
    check_if_equal(proj_data2.get_viewgram(1,1).find_max(),
                   proj_data.get_viewgram(1,1).find_max(),
                   "test 1 for copy-constructor and get_viewgram");
    proj_data2.fill(1e4f);
    check(std::abs(proj_data2.get_viewgram(0,0).find_max() -
          proj_data.get_viewgram(0,0).find_max()) > 1.f,
          "test 1 for deep copy and get_viewgram");
  }

  // test fill with larger input
  {    
    shared_ptr<ProjDataInfo> proj_data_info_sptr2
      (ProjDataInfo::ProjDataInfoCTI(scanner_sptr,
                                     /*span*/1, 8,/*views*/ 96, /*tang_pos*/128, /*arc_corrected*/ true)
       );
  
      
    // construct without filling
    ProjDataInMemory proj_data2(exam_info_sptr, proj_data_info_sptr2, false);
    proj_data2.fill(proj_data);
    check_if_equal(proj_data2.get_viewgram(0,0).find_max(),
                   proj_data.get_viewgram(0,0).find_max(),
                   "test 1 for copy-constructor and get_viewgram");
    check_if_equal(proj_data2.get_viewgram(1,1).find_max(),
                   proj_data.get_viewgram(1,1).find_max(),
                   "test 1 for copy-constructor and get_viewgram");
  }

  // test fill with smaller input
  {    
    shared_ptr<ProjDataInfo> proj_data_info_sptr2 
      (ProjDataInfo::ProjDataInfoCTI(scanner_sptr,
                                     /*span*/1, 12,/*views*/ 96, /*tang_pos*/128, /*arc_corrected*/ true)
       );
  
      
    // construct without filling
    ProjDataInMemory proj_data2(exam_info_sptr, proj_data_info_sptr2, false);
    // this should call error, so we'll catch it
    try
      {
        std::cout << "\nthis test should intentionally throw an error\n";
        proj_data2.fill(proj_data);
        check(false, "test fill wtih too small proj_data should have thrown");
      }
    catch (...)
      {
        // ok
      }
  }
//  test set_bin_value() and get_bin_value
  {
      std::vector<float> test;
      test.resize(proj_data.size_all());
      
      for(int i=0;i<test.size();i++)
      test[i]=i;
      
      proj_data.fill_from(test.begin());
      
      Bin bin(0,proj_data_info_sptr->get_max_view_num()/2,
                proj_data_info_sptr->get_max_axial_pos_num(0)/2,
                0);
      
      bin.set_bin_value(42);
      proj_data.set_bin_value(bin);
      check_if_equal(bin.get_bin_value(),proj_data.get_bin_value(bin),
            "ProjDataInMemory::set_bin_value/get_bin_value not consistent");
      // also check via get_viewgram
      const Viewgram<float> viewgram=proj_data.get_viewgram(bin.view_num(), bin.segment_num());
      check_if_equal(bin.get_bin_value(),viewgram[bin.axial_pos_num()][bin.tangential_pos_num()],
            "ProjDataInMemory::set_bin_value/get_viewgram not consistent");
  }
}

END_NAMESPACE_STIR


USING_NAMESPACE_STIR

int main()
{
  ProjDataInMemoryTests tests;
  tests.run_tests();
  return tests.main_return_value();
}
