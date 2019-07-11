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
//  std::cout << "-------- Testing Upsampling and Downsampling ---------\n";

    std::cout << "DONE" <<'\n';




}

END_NAMESPACE_STIR


USING_NAMESPACE_STIR

int main()
{
  SSRBTests tests;
  tests.run_tests();
  return tests.main_return_value();
}
