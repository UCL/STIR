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
void check_proj_data_are_equal(const ProjData& x, const ProjData& y)
{
    const size_t n = x.size_all();
    const size_t ny = y.size_all();
    if (n!=ny)
        error("ProjData::axpby and ProjDataInMemory::axpby mismatch");

    // Create arrays
    std::vector<float> arr1(n), arr2(n);
    x.copy_to(arr1.begin());
    y.copy_to(arr2.begin());

    for (unsigned i=0; i<n; ++i)
        if (std::abs(arr1[i]-arr2[i]) > 1e-4f)
            error("ProjData::axpby and ProjDataInMemory::axpby mismatch");
}

void
ProjDataInMemoryTests::
run_tests()
{
    // Create scanner and proj data info
    shared_ptr<Scanner> scanner_sptr(new Scanner(Scanner::E953));
    shared_ptr<ProjDataInfo> proj_data_info_sptr
            (ProjDataInfo::ProjDataInfoCTI(scanner_sptr,
             /*span*/1, 10,/*views*/ 96, /*tang_pos*/128, /*arc_corrected*/ true));

    // Create and fill
    shared_ptr<ExamInfo> exam_info_sptr(new ExamInfo);
    ProjDataInMemory pd1(exam_info_sptr, proj_data_info_sptr);
    const float value = 1.2F;
    pd1.fill(value);

    // Copy
    ProjDataInMemory pd2(pd1);

    // Check axpby with general and ProjDataInMemory methods
    pd1.axpby(2.f,pd1,3.f,pd1);
    pd2.ProjData::axpby(2.f,pd2,3.f,pd2);

    check_proj_data_are_equal(pd1,pd2);
}

END_NAMESPACE_STIR


USING_NAMESPACE_STIR

int main()
{
  ProjDataInMemoryTests tests;
  tests.run_tests();
  return tests.main_return_value();
}
