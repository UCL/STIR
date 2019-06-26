//
//
/*!

  \file
  \ingroup test

  \author Ludovica Brusaferri

*/
/*
    Copyright (C) 2015, University College London
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
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/scatter/SingleScatterSimulation.h"
#include "stir/scatter/ScatterEstimation.h"


START_NAMESPACE_STIR


class UpsampleDownsampleTests: public RunTests
{   
public:
  void run_tests();
};

void
UpsampleDownsampleTests::
run_tests()
{
  std::cout << "-------- Testing Upsampling and Downsampling --------\n";

  //creating scanner shared pointer
  shared_ptr<Scanner> scanner_sptr(new Scanner(Scanner::E953));

  //creating proj data info
  shared_ptr<ProjDataInfo> proj_data_info_sptr
    (ProjDataInfo::ProjDataInfoCTI(scanner_sptr,
                            /*span*/1, 1,/*views*/ 96, /*tang_pos*/128, /*arc_corrected*/ true)
     );

  //creating low resolution proj data info
  shared_ptr<ProjDataInfo> LR_proj_data_info_sptr
    (ProjDataInfo::ProjDataInfoCTI(scanner_sptr,
                            /*span*/1, 1,/*views*/ 24, /*tang_pos*/32, /*arc_corrected*/ true)
     );
  shared_ptr<ExamInfo> exam_info_sptr(new ExamInfo);


  // construct projdata with filling to 0
  ProjDataInMemory proj_data(exam_info_sptr, proj_data_info_sptr);
  {
    Sinogram<float> sinogram = proj_data.get_sinogram(0,0);
    check_if_equal(sinogram.find_min(),
                   0.F,
                   "test constructor and get_sinogram");
  }

  //std::cout << "TEST: 3D out size:"<<  proj_data.get_num_segments() << "x"<< proj_data.get_segment_by_sinogram(0).get_num_views()<< "x" <<  proj_data.get_segment_by_sinogram(0).get_num_tangential_poss()<< '\n';

  proj_data.fill(1);

  // construct projdata with filling to 5
  ProjDataInMemory proj_data_LR(exam_info_sptr, LR_proj_data_info_sptr);
  {
    Sinogram<float> sinogram_LR = proj_data_LR.get_sinogram(0,0);
    check_if_equal(sinogram_LR.find_min(),
                   0.F,
                   "test constructor and get_sinogram");
  }
  proj_data_LR.fill(1);

  //std::cout << "TEST: 3D LR size:"<<  proj_data_LR.get_num_segments() << "x"<< proj_data_LR.get_segment_by_sinogram(0).get_num_views()<< "x" <<  proj_data_LR.get_segment_by_sinogram(0).get_num_tangential_poss()<< '\n';


 ProjDataInMemory scaled_proj_data = proj_data;
 scaled_proj_data.fill(1);

 //std::cout << "TEST: 3D scaled size:"<<  scaled_proj_data.get_num_segments() << "x"<< scaled_proj_data.get_segment_by_sinogram(0).get_num_views()<< "x" <<  scaled_proj_data.get_segment_by_sinogram(0).get_num_tangential_poss()<< '\n';

 std::cout << "-------- Testing Pull --------\n";
 ScatterEstimation::pull_scatter_estimate(scaled_proj_data,proj_data,proj_data_LR,false);
    std::cout << "-------- Testing Push --------\n";

 ProjDataInMemory scaled_proj_data_LR = proj_data_LR;
 scaled_proj_data_LR.fill(1);
 ScatterEstimation::push_scatter_estimate(scaled_proj_data_LR,proj_data_LR,scaled_proj_data,false);

 //std::cout << "TEST: 3D downsampled size:"<<  scaled_proj_data_LR.get_num_segments() << "x"<< scaled_proj_data_LR.get_segment_by_sinogram(0).get_num_views()<< "x" <<  scaled_proj_data_LR.get_segment_by_sinogram(0).get_num_tangential_poss()<< '\n';

}

END_NAMESPACE_STIR


USING_NAMESPACE_STIR

int main()
{
  UpsampleDownsampleTests tests;
  tests.run_tests();
  return tests.main_return_value();
}
