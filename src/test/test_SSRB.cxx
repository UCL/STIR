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
#include "stir/inverse_SSRB.h"
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


class SSRBTests: public RunTests
{   
public:
  void run_tests();
  void fill_projdata_with_random(ProjData & projdata);
};

void
SSRBTests::
fill_projdata_with_random(ProjData & projdata)
{
    Bin bin;
    {
        for (bin.segment_num()=projdata.get_min_segment_num();
             bin.segment_num()<=projdata.get_max_segment_num();
             ++bin.segment_num())
            for (bin.axial_pos_num()=
                 projdata.get_min_axial_pos_num(bin.segment_num());
                 bin.axial_pos_num()<=projdata.get_max_axial_pos_num(bin.segment_num());
                 ++bin.axial_pos_num())
            {

                Sinogram<float> sino = projdata.get_empty_sinogram(bin.axial_pos_num(),bin.segment_num());

                for (bin.view_num()=sino.get_min_view_num();
                     bin.view_num()<=sino.get_max_view_num();
                     ++bin.view_num())
                {
                    for (bin.tangential_pos_num()=
                         sino.get_min_tangential_pos_num();
                         bin.tangential_pos_num()<=
                         sino.get_max_tangential_pos_num();
                         ++bin.tangential_pos_num())
                         sino[bin.view_num()][bin.tangential_pos_num()]= rand()%10;//((double) rand() / (1)) + 1;

                    projdata.set_sinogram(sino);

                 }

              }

        }
}
void
SSRBTests::
run_tests()
{


    std::cout << "========== TEST SSRB =========== \n";

    shared_ptr<Scanner> scanner_sptr(new Scanner(Scanner::Siemens_mMR));

    //creating proj data info
    shared_ptr<ProjDataInfo> proj_data_info_sptr_4D(ProjDataInfo::ProjDataInfoCTI(scanner_sptr,/*span*/1, 1,/*views*/ 252, /*tang_pos*/344, /*arc_corrected*/ false));
    shared_ptr<ExamInfo> exam_info_sptr_4D(new ExamInfo);
    ProjDataInMemory projdata_4D(exam_info_sptr_4D, proj_data_info_sptr_4D);

    shared_ptr<ProjDataInfo> proj_data_info_sptr_3D(projdata_4D.get_proj_data_info_ptr()->clone());
    proj_data_info_sptr_3D->reduce_segment_range(0,0); //create input template

    ProjDataInMemory projdata_3D(projdata_4D.get_exam_info_sptr(),proj_data_info_sptr_3D);

    ProjDataInMemory A_4D(exam_info_sptr_4D, proj_data_info_sptr_4D);
    ProjDataInMemory A_3D(projdata_4D.get_exam_info_sptr(),proj_data_info_sptr_3D);

    fill_projdata_with_random(projdata_4D);
    fill_projdata_with_random(projdata_3D);
    A_4D.fill(0); //initialise output
    A_3D.fill(0); //initialise output


    std::cout << "-------- Testing Inverse SSRB --------\n";
    inverse_SSRB(A_4D,projdata_3D);

    std::cout << "-------- Testing Transpose Inverse SSRB --------\n";

    transpose_inverse_SSRB(A_3D,projdata_4D);

    std::cout << "-------- <Ax|y> = <x|A*y> --------\n";

    float cdot1 = 0;
    float cdot2 = 0;

    for ( int segment_num = projdata_4D.get_min_segment_num(); segment_num <= projdata_4D.get_max_segment_num(); ++segment_num)
      {
        for (int axial_pos = projdata_4D.get_min_axial_pos_num(segment_num); axial_pos <= projdata_4D.get_max_axial_pos_num(segment_num); ++axial_pos)
          {

            const Sinogram<float> yy_sinogram = projdata_4D.get_sinogram(axial_pos, segment_num);
            const Sinogram<float> At_sinogram = A_4D.get_sinogram(axial_pos, segment_num);

            for ( int view_num = projdata_4D.get_min_view_num(); view_num <= projdata_4D.get_max_view_num();view_num++)
              {
                for ( int tang_pos = projdata_4D.get_min_tangential_pos_num(); tang_pos <= projdata_4D.get_max_tangential_pos_num(); tang_pos++)
                 {
                     cdot1 += At_sinogram[view_num][tang_pos]*yy_sinogram[view_num][tang_pos];
                  }
               }
             }
          }

    for ( int segment_num = projdata_3D.get_min_segment_num(); segment_num <= projdata_3D.get_max_segment_num(); ++segment_num)
      {
        for (int axial_pos = projdata_3D.get_min_axial_pos_num(segment_num); axial_pos <= projdata_3D.get_max_axial_pos_num(segment_num); ++axial_pos)
          {

            const Sinogram<float> xx_sinogram = projdata_3D.get_sinogram(axial_pos, segment_num);
            const Sinogram<float> A_sinogram = A_3D.get_sinogram(axial_pos, segment_num);

            for ( int view_num = projdata_3D.get_min_view_num(); view_num <= projdata_3D.get_max_view_num();view_num++)
              {
                for ( int tang_pos = projdata_3D.get_min_tangential_pos_num(); tang_pos <= projdata_3D.get_max_tangential_pos_num(); tang_pos++)
                 {
                     cdot2 += xx_sinogram[view_num][tang_pos]*A_sinogram[view_num][tang_pos];
                  }
               }
             }
          }

    std::cout << cdot1 << "=" << cdot2 << '\n';
    set_tolerance(0.02);
    check_if_equal(cdot1, cdot2, "test adjoint");




}

END_NAMESPACE_STIR


USING_NAMESPACE_STIR

int main()
{
  SSRBTests tests;
  tests.run_tests();
  return tests.main_return_value();
}
