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
#include "stir/Sinogram.h"
#include "stir/Array.h"

#include "stir/IO/write_to_file.h"



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
  std::cout << "-------- Testing Upsampling and Downsampling ---------\n";

    shared_ptr<Scanner> scanner_sptr(new Scanner(Scanner::Siemens_mMR));
    shared_ptr<ExamInfo> exam_info_sptr(new ExamInfo);
    shared_ptr<ExamInfo> LR_exam_info_sptr(new ExamInfo);

    //creating proj data info
    shared_ptr<ProjDataInfo> proj_data_info_sptr(ProjDataInfo::ProjDataInfoCTI(scanner_sptr,/*span*/1, 0,/*views*/ 252, /*tang_pos*/344, /*arc_corrected*/ false));

    shared_ptr<ProjData> LR = ProjData::read_from_file("simulated_scatter_sino_UU2.hs");

    // construct y
    ProjDataInMemory y(exam_info_sptr, proj_data_info_sptr);

    // construct x
    ProjDataInMemory x(*LR);

    // construct Ax

    ProjDataInMemory Ax(exam_info_sptr, proj_data_info_sptr);
    // construct x
    ProjDataInMemory Aty(*LR);

    x.fill(1);
    y.fill(2);

    Ax.fill(0); //initialise output
    Aty.fill(0); //initialise output

    bool remove_interleaving = false;

    std::cout << "========== REMOVE INTERLEAVING = FALSE =========== \n";

    std::cout << "-------- Testing Pull --------\n";
    ScatterEstimation::pull_scatter_estimate(Ax,y,x,remove_interleaving);

    std::cout << "-------- Testing Push --------\n";

    ScatterEstimation::push_scatter_estimate(Aty,x,y,remove_interleaving);

    std::cout << "-------- <Ax|y> = <x|A*y> --------\n";

    float cdot1 = 0;
    float cdot2 = 0;

    for ( int segment_num = y.get_min_segment_num(); segment_num <= y.get_max_segment_num(); ++segment_num)
      {
        for (int axial_pos = y.get_min_axial_pos_num(segment_num); axial_pos <= y.get_max_axial_pos_num(segment_num); ++axial_pos)
          {

            const Sinogram<float> y_sinogram = y.get_sinogram(axial_pos, segment_num);
            const Sinogram<float> Ax_sinogram = Ax.get_sinogram(axial_pos, segment_num);

            for ( int view_num = y.get_min_view_num(); view_num <= y.get_max_view_num();view_num++)
              {
                for ( int tang_pos = y.get_min_tangential_pos_num(); tang_pos <= y.get_max_tangential_pos_num(); tang_pos++)
                 {
                     cdot1 += Ax_sinogram[view_num][tang_pos]*y_sinogram[view_num][tang_pos];
                  }
               }
             }
          }

    for ( int segment_num = x.get_min_segment_num(); segment_num <= x.get_max_segment_num(); ++segment_num)
      {
        for (int axial_pos = x.get_min_axial_pos_num(segment_num); axial_pos <= x.get_max_axial_pos_num(segment_num); ++axial_pos)
          {

            const Sinogram<float> x_sinogram = x.get_sinogram(axial_pos, segment_num);
            const Sinogram<float> Aty_sinogram = Aty.get_sinogram(axial_pos, segment_num);

            for ( int view_num = x.get_min_view_num(); view_num <= x.get_max_view_num();view_num++)
              {
                for ( int tang_pos = x.get_min_tangential_pos_num(); tang_pos <= x.get_max_tangential_pos_num(); tang_pos++)
                 {
                     cdot2 += x_sinogram[view_num][tang_pos]*Aty_sinogram[view_num][tang_pos];
                  }
               }
             }
          }

    std::cout << cdot1 << "=" << cdot2 << '\n';
    check_if_equal(cdot1, cdot2, "test adjoint");


    std::cout << "========== REMOVE INTERLEAVING = TRUE =========== \n";

    Ax.fill(0); //initialise output
    Aty.fill(0); //initialise output

    remove_interleaving = true;

    std::cout << "-------- Testing Pull --------\n";
    ScatterEstimation::pull_scatter_estimate(Ax,y,x,remove_interleaving);

    std::cout << "-------- Testing Push --------\n";

    ScatterEstimation::push_scatter_estimate(Aty,x,y,remove_interleaving);

    std::cout << "-------- <Ax|y> = <x|A*y> --------\n";

    cdot1 = 0;
    cdot2 = 0;

    for ( int segment_num = y.get_min_segment_num(); segment_num <= y.get_max_segment_num(); ++segment_num)
      {
        for (int axial_pos = y.get_min_axial_pos_num(segment_num); axial_pos <= y.get_max_axial_pos_num(segment_num); ++axial_pos)
          {

            const Sinogram<float> y_sinogram = y.get_sinogram(axial_pos, segment_num);
            const Sinogram<float> Ax_sinogram = Ax.get_sinogram(axial_pos, segment_num);

            for ( int view_num = y.get_min_view_num(); view_num <= y.get_max_view_num();view_num++)
              {
                for ( int tang_pos = y.get_min_tangential_pos_num(); tang_pos <= y.get_max_tangential_pos_num(); tang_pos++)
                 {
                     cdot1 += Ax_sinogram[view_num][tang_pos]*y_sinogram[view_num][tang_pos];
                  }
               }
             }
          }

    for ( int segment_num = x.get_min_segment_num(); segment_num <= x.get_max_segment_num(); ++segment_num)
      {
        for (int axial_pos = x.get_min_axial_pos_num(segment_num); axial_pos <= x.get_max_axial_pos_num(segment_num); ++axial_pos)
          {

            const Sinogram<float> x_sinogram = x.get_sinogram(axial_pos, segment_num);
            const Sinogram<float> Aty_sinogram = Aty.get_sinogram(axial_pos, segment_num);

            for ( int view_num = x.get_min_view_num(); view_num <= x.get_max_view_num();view_num++)
              {
                for ( int tang_pos = x.get_min_tangential_pos_num(); tang_pos <= x.get_max_tangential_pos_num(); tang_pos++)
                 {
                     cdot2 += x_sinogram[view_num][tang_pos]*Aty_sinogram[view_num][tang_pos];
                  }
               }
             }
          }

    std::cout << cdot1 << "=" << cdot2 << '\n';
    check_if_equal(cdot1, cdot2, "test adjoint");



}

END_NAMESPACE_STIR


USING_NAMESPACE_STIR

int main()
{
  UpsampleDownsampleTests tests;
  tests.run_tests();
  return tests.main_return_value();
}
