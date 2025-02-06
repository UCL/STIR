//
//
/*!

  \file
  \ingroup test
  \ingroup projdata

  \brief Test maths of stir::ProjData

  \author Richard Brown
  \author Kris Thielemans

*/
/*
    Copyright (C) 2020, 2024 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

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
#include "stir/copy_fill.h"
#include "stir/error.h"
#include <string>
START_NAMESPACE_STIR

/*!
  \ingroup test
  \ingroup projdata
  \brief Test class for ProjDataInMemory
*/
class ProjDataInMemoryTests : public RunTests
{
public:
  void run_tests() override;

private:
  void run_tests(shared_ptr<const ExamInfo> exam_info_sptr, shared_ptr<const ProjDataInfo> proj_data_info_sptr);
};

static void
check_proj_data_are_equal_and_non_zero(const ProjData& x, const ProjData& y, const std::string& name)
{
  const size_t n = x.size_all();
  const size_t ny = y.size_all();
  if (n != ny)
    error("ProjData::" + name + " and ProjDataInMemory::" + name + " mismatch");

  // Create arrays
  std::vector<float> arr1(n), arr2(n);
  copy_to(x, arr1.begin());
  copy_to(y, arr2.begin());

  // Check for mismatch
  for (unsigned i = 0; i < n; ++i)
    if (std::abs(arr1[i] - arr2[i]) > 1e-4f)
      error("ProjData::" + name + " and ProjDataInMemory::" + name + " mismatch");

  // Check for non-zero
  if (std::abs(*std::max_element(arr1.begin(), arr1.end())) < 1e-4f)
    error("ProjData::" + name + " and ProjDataInMemory::" + name + " mismatch");
}

void
ProjDataInMemoryTests::run_tests(shared_ptr<const ExamInfo> exam_info_sptr, shared_ptr<const ProjDataInfo> proj_data_info_sptr)
{
  ProjDataInMemory pd1(exam_info_sptr, proj_data_info_sptr);
  ProjDataInMemory pd2(pd1);

  // Create x1 and x2
  ProjDataInMemory x1(pd1);
  x1.fill(100.f);
  ProjDataInMemory x2(x1);

  // Create y1 and y2
  ProjDataInMemory y1(pd1);
  y1.fill(1000.f);
  ProjDataInMemory y2(y1);

  // Check xapby with general and ProjDataInMemory methods
  const float a = 2.f;
  const float b = 3.f;
  pd1.xapyb(x1, a, y1, b);
  pd2.ProjData::xapyb(x2, a, y2, b);
  check_proj_data_are_equal_and_non_zero(pd1, pd2, "xapyb");

  // Check sapby with general and ProjDataInMemory methods
  ProjDataInMemory out1(x1);
  out1.sapyb(a, y1, b);
  check_proj_data_are_equal_and_non_zero(pd1, out1, "sapyb");

  ProjDataInMemory out2(x1);
  out2.ProjData::sapyb(a, y1, b);
  check_proj_data_are_equal_and_non_zero(pd1, out2, "sapyb");

  // Check using iterators
  ProjDataInMemory pd3(pd1);
  pd3.fill(0.f);
  ProjDataInMemory::iterator pd_iter = pd3.begin();
  ProjDataInMemory::const_iterator x_iter = x1.begin();
  ProjDataInMemory::const_iterator y_iter = y1.begin();
  while (pd_iter != pd3.end())
    *pd_iter++ = a * (*x_iter++) + b * (*y_iter++);

  check_proj_data_are_equal_and_non_zero(pd1, pd3, "fill");

  // clang-format 14.0 makes a complete mess of the stuff below, so we'll switch if off
  // clang-format off

  // numeric operations
  {
    {
      auto res1 = pd1 + pd2;
      {
        ProjDataInMemory res2(pd1);
        res2.sapyb(1.F, pd2, 1.F);
        check_if_equal(res1, res2, "+ vs sapyb");
      }
      {
        ProjDataInMemory res2(pd1);
        res2 += pd2;
        check_if_equal(res1, res2, "+ vs +=");
      }
      {
        check_if_equal(norm(pd1 + pd1), 2 * norm(pd1), "norm of x+x");
      }
    }

    {
      auto res1 = pd1 - pd2;
      {
        ProjDataInMemory res2(pd1);
        res2.sapyb(1.F, pd2, -1.F);
        check_if_equal(res1, res2, "- vs sapyb");
      }
      {
        ProjDataInMemory res2(pd1);
        res2 -= pd2;
        check_if_equal(res1, res2, "- vs -=");
      }
      {
        res1 += pd2;
        check_if_equal(pd1, res1, "- vs +=");
      }
      {
        check_if_zero(norm(pd1 - pd1), "norm of x-x");
      }
    }
    {
      auto res1 = pd1 * pd2;
      {
        ProjDataInMemory res2(pd1);
        for (auto i1 = res2.begin(), i2 = pd2.begin(); i1 != res2.end(); ++i1, ++i2)
          *i1 *= *i2;
        check_if_equal(res1, res2, "* vs loop");
      }
      {
        ProjDataInMemory res2(pd1);
        res2 *= pd2;
        check_if_equal(res1, res2, "* vs *=");
      }
      {
        res1 /= pd2;
        check_if_equal(pd1, res1, "* vs /=");
      }
    }
    {
      auto res1 = pd1 / pd2;
      {
        ProjDataInMemory res2(pd1);
        for (auto i1 = res2.begin(), i2 = pd2.begin(); i1 != res2.end(); ++i1, ++i2)
          *i1 /= *i2;
        check_if_equal(res1, res2, "/ vs loop");
      }
      {
        ProjDataInMemory res2(pd1);
        res2 /= pd2;
        check_if_equal(res1, res2, "/ vs /=");
      }
      {
        res1 *= pd2;
        check_if_equal(pd1, res1, "/ vs *=");
      }
      {
        // assumes that all elements are !=0
        check_if_equal(norm_squared(pd1 / pd1), static_cast<double>(pd1.size_all()), "norm of x/x");
      }
    }
    // now with floats
    {
      auto res1 = pd1 + 5.6F;
      check_if_equal(res1.find_max(), pd1.find_max() + 5.6F, "max(x + 5.6F)");
      {
        ProjDataInMemory res2(pd1);
        res2 += 5.6F;
        check_if_equal(res1, res2, "+ vs += float");
      }
      {
        res1 -= pd1;
        res1 /= 5.6F;
        check_if_equal(norm_squared(res1), static_cast<double>(pd1.size_all()), "norm of x + 5.6");
      }
    }
    {
      auto res1 = pd1 - 5.6F;
      check_if_equal(res1.find_min(), pd1.find_min() - 5.6F, "min(x - 5.6F)");
      {
        ProjDataInMemory res2(pd1);
        res2 -= 5.6F;
        check_if_equal(res1, res2, "- vs -= float");
      }
      {
        res1 += 5.6F;
        check_if_equal(res1, pd1, "- vs += float");
      }
    }
    {
      auto res1 = pd1 * 5.6F;
      check_if_equal(norm(res1), norm(pd1) * 5.6F, "norm of x*5.6");
      check_if_equal(res1.sum(), pd1.sum() * 5.6F, "sum(x * 5.6F)");

      {
        ProjDataInMemory res2(pd1);
        res2 *= 5.6F;
        check_if_equal(res1, res2, "* vs *= float");
      }
      {
        res1 /= 5.6F;
        check_if_equal(res1, pd1, "* vs /= float");
      }
    }
    {
      auto res1 = pd1 / 5.6F;
      check_if_equal(norm(res1), norm(pd1) / 5.6F, "norm of x/float");

      {
        ProjDataInMemory res2(pd1);
        res2 /= 5.6F;
        check_if_equal(res1, res2, "/ vs /= float");
      }
      {
        res1 /= 1 / 5.6F;
        check_if_equal(res1, pd1, "/ vs /= float 2");
      }
    }
  }

  // numeric operations ProjDataInMemory vs ProjData (check if inverse operation)
  {
    ProjDataInMemory pd1_copy(pd1);
    // with ProjData arg
    {
      pd1_copy += pd2;
      pd1_copy.ProjData::operator-=(pd2);
      check_if_equal(pd1, pd1_copy, "ProjDataInMemory::+=(ProjData&) vs ProjData::-=");
    }
    {
      pd1_copy -= pd2;
      pd1_copy.ProjData::operator+=(pd2);
      check_if_equal(pd1, pd1_copy, "ProjDataInMemory::-=(ProjData&) vs ProjData::+=");
    }
    {
      pd1_copy *= pd2;
      pd1_copy.ProjData::operator/=(pd2);
      check_if_equal(pd1, pd1_copy, "ProjDataInMemory::*=(ProjData&) vs ProjData::/=");
    }
    {
      pd1_copy /= pd2;
      pd1_copy.ProjData::operator*=(pd2);
      check_if_equal(pd1, pd1_copy, "ProjDataInMemory::/=(ProjData&) vs ProjData::*=");
    }
    // with float arg
    const float arg = 5.9F;
    {
      pd1_copy += arg;
      pd1_copy.ProjData::operator-=(arg);
      check_if_equal(pd1, pd1_copy, "ProjDataInMemory::+=(float) vs ProjData::-=");
    }
    {
      pd1_copy -= arg;
      pd1_copy.ProjData::operator+=(arg);
      check_if_equal(pd1, pd1_copy, "ProjDataInMemory::-=(float) vs ProjData::+=");
    }
    {
      pd1_copy *= arg;
      pd1_copy.ProjData::operator/=(arg);
      check_if_equal(pd1, pd1_copy, "ProjDataInMemory::*=(float) vs ProjData::/=");
    }
    {
      pd1_copy /= arg;
      pd1_copy.ProjData::operator*=(arg);
      check_if_equal(pd1, pd1_copy, "ProjDataInMemory::/=(float) vs ProjData::*=");
    }
  }
// clang-format on
}

void
ProjDataInMemoryTests::run_tests()
{
  std::cerr << "tests on proj_data maths\n";

  std::cerr << "------------------ non-TOF\n";
  {
    // Create scanner and proj data info
    shared_ptr<Scanner> scanner_sptr(new Scanner(Scanner::E953));
    shared_ptr<ProjDataInfo> proj_data_info_sptr(ProjDataInfo::construct_proj_data_info(scanner_sptr,
                                                                                        /*span*/ 1,
                                                                                        10,
                                                                                        /*views*/ 96,
                                                                                        /*tang_pos*/ 128,
                                                                                        /*arc_corrected*/ true));

    // Create pd1 and pd2
    shared_ptr<ExamInfo> exam_info_sptr(new ExamInfo);
    exam_info_sptr->imaging_modality = ImagingModality::PT;

    run_tests(exam_info_sptr, proj_data_info_sptr);
  }

  std::cerr << "------------------ TOF\n";
  {
    // Create scanner and proj data info
    shared_ptr<Scanner> scanner_sptr(new Scanner(Scanner::Discovery690));

    shared_ptr<ProjDataInfo> proj_data_info_sptr(
        ProjDataInfo::construct_proj_data_info(scanner_sptr,
                                               /*span*/ 2,
                                               5,
                                               /*views*/ scanner_sptr->get_num_detectors_per_ring() / 4,
                                               /*tang_pos*/ 22,
                                               /*arc_corrected*/ false,
                                               /* Tof_mashing */ 11));

    // Create pd1 and pd2
    shared_ptr<ExamInfo> exam_info_sptr(new ExamInfo);
    exam_info_sptr->imaging_modality = ImagingModality::PT;

    run_tests(exam_info_sptr, proj_data_info_sptr);
  }
}

END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int
main()
{
  ProjDataInMemoryTests tests;
  tests.run_tests();
  return tests.main_return_value();
}
