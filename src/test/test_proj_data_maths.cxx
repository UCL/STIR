//
//
/*!

  \file
  \ingroup test

  \brief Test maths of stir::ProjData

  \author Richard Brown

*/
/*
    Copyright (C) 2020 University College London
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

static
void check_proj_data_are_equal_and_non_zero(const ProjData& x, const ProjData& y)
{
    const size_t n = x.size_all();
    const size_t ny = y.size_all();
    if (n!=ny)
        error("ProjData::axpby and ProjDataInMemory::axpby mismatch");

    // Create arrays
    std::vector<float> arr1(n), arr2(n);
    x.copy_to(arr1.begin());
    y.copy_to(arr2.begin());

    // Check for mismatch
    for (unsigned i=0; i<n; ++i)
        if (std::abs(arr1[i]-arr2[i]) > 1e-4f)
            error("ProjData::axpby and ProjDataInMemory::axpby mismatch");

    // Check for non-zero
    if (std::abs(*std::max_element(arr1.begin(),arr1.end())) < 1e-4f)
        error("ProjData::axpby and ProjDataInMemory::axpby mismatch");
}

static void fill(ProjData &pd, const float start_val)
{
    float val = start_val;
    const int n_min = pd.get_min_segment_num();
    const int n_max = pd.get_max_segment_num();
    for (int s=n_min; s<=n_max; ++s) {
        SegmentBySinogram<float> seg = pd.get_empty_segment_by_sinogram(s);
        SegmentBySinogram<float>::full_iterator seg_iter;
        for (seg_iter = seg.begin_all();
             seg_iter != seg.end_all();
             ++seg_iter, ++val)
        {
            *seg_iter = val;
        }
        pd.set_segment(seg);
    }
}

void
ProjDataInMemoryTests::
run_tests()
{
    // Create scanner and proj data info
    shared_ptr<Scanner> scanner_sptr(new Scanner(Scanner::E953));
    shared_ptr<ProjDataInfo>
      proj_data_info_sptr(
            ProjDataInfo::construct_proj_data_info
            (scanner_sptr,
             /*span*/1, 10,/*views*/ 96, /*tang_pos*/128, /*arc_corrected*/ true)
      );

    // Create pd1 and pd2
    shared_ptr<ExamInfo> exam_info_sptr(new ExamInfo);
    ProjDataInMemory pd1(exam_info_sptr, proj_data_info_sptr);
    ProjDataInMemory pd2(pd1);

    // Create x1 and x2
    ProjDataInMemory x1(pd1);
    fill(x1,100.f);
    ProjDataInMemory x2(x1);

    // Create y1 and y2
    ProjDataInMemory y1(pd1);
    fill(y1,1000.f);
    ProjDataInMemory y2(y1);

    // Check axpby with general and ProjDataInMemory methods
    const float a = 2.f;
    const float b = 3.f;
    pd1.axpby(a,x1,b,y1);
    pd2.ProjData::axpby(a,x2,b,y2);

    check_proj_data_are_equal_and_non_zero(pd1,pd2);

    // Check using iterators
    ProjDataInMemory pd3(pd1);
    pd3.fill(0.f);
    ProjDataInMemory::iterator pd_iter = pd3.begin();
    ProjDataInMemory::const_iterator x_iter = x1.begin();
    ProjDataInMemory::const_iterator y_iter = y1.begin();
    while (pd_iter != pd3.end())
        *pd_iter++ = a*(*x_iter++) + b*(*y_iter++);

    check_proj_data_are_equal_and_non_zero(pd1,pd3);
}

END_NAMESPACE_STIR


USING_NAMESPACE_STIR

int main()
{
  ProjDataInMemoryTests tests;
  tests.run_tests();
  return tests.main_return_value();
}
