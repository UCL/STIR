/*!

  \file
  \ingroup test

  \brief Test program for stir::ProjData and stir::ProjDataInMemory

  \author Kris Thielemans
  \author Daniel Deidda

*/
/*
    Copyright (C) 2015, 2020, 2022 University College London
    Copyright (C) 2020, National Physical Laboratory
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include "stir/ProjDataInMemory.h"
#include "stir/ProjDataInterfile.h"
#include "stir/ExamInfo.h"
#include "stir/ProjDataInfo.h"
#include "stir/ProjDataInfoCylindricalArcCorr.h"
#include "stir/Sinogram.h"
#include "stir/Viewgram.h"
#include "stir/Succeeded.h"
#include "stir/RunTests.h"
#include "stir/Scanner.h"
#include "stir/copy_fill.h"
#include "stir/numerics/norm.h"
#include "stir/IndexRange4D.h"
#include "stir/CPUTimer.h"
#include <algorithm>
#include <numeric>

START_NAMESPACE_STIR

/*!
  \ingroup test
  \brief Test class for ProjData and ProjDataInMemory
*/
class ProjDataTests : public RunTests
{
public:
  void run_tests() override;

private:
  void run_tests_on_proj_data(ProjData&);
  void run_tests_in_memory_only(ProjDataInMemory&);
};

void
ProjDataTests::run_tests_on_proj_data(ProjData& proj_data)
{

  CPUTimer timer;
  const float value = 1.2F;
  std::cerr << "test fill(float)\n";
  timer.reset();
  timer.start();
  {
    proj_data.fill(value);
    Viewgram<float> viewgram = proj_data.get_viewgram(0, 0);
    check_if_equal(viewgram.find_min(), value, "test fill(float) and get_viewgram");
  }

  timer.stop();
  std::cerr << "-- CPU Time " << timer.value() << '\n';
  std::cerr << "\ntest set_viewgram\n";
  timer.reset();
  timer.start();
  {
    Viewgram<float> viewgram = proj_data.get_empty_viewgram(1, 1);
    viewgram.fill(value * 2);
    check(proj_data.set_viewgram(viewgram) == Succeeded::yes, "test set_viewgram succeeded");

    Viewgram<float> viewgram2 = proj_data.get_viewgram(1, 1);
    check_if_equal(viewgram2.find_min(), viewgram.find_min(), "test set/get_viewgram");
  }

  timer.stop();
  std::cerr << "-- CPU Time " << timer.value() << '\n';
  std::cerr << "\ntest making a copy to ProjDataInMemory\n";
  timer.reset();
  timer.start();
  {
    ProjDataInMemory proj_data2(proj_data);
    check_if_equal(proj_data2.get_viewgram(0, 0).find_max(),
                   proj_data.get_viewgram(0, 0).find_max(),
                   "test 1 for copy-constructor and get_viewgram");
    check_if_equal(proj_data2.get_viewgram(1, 1).find_max(),
                   proj_data.get_viewgram(1, 1).find_max(),
                   "test 1 for copy-constructor and get_viewgram");
    // check this was a deep-copy by filling and check if the original is not affected
    proj_data2.fill(1e4f);
    check(std::abs(proj_data2.get_viewgram(0, 0).find_max() - proj_data.get_viewgram(0, 0).find_max()) > 1.f,
          "test 1 for deep copy and get_viewgram");
  }
  timer.stop();
  std::cerr << "-- CPU Time " << timer.value() << '\n';

  std::cerr << "\ntest fill(ProjDataInMemory)\n";
  timer.reset();
  timer.start();
  {
    ProjDataInMemory proj_data2(proj_data.get_exam_info_sptr(), proj_data.get_proj_data_info_sptr());
    // fill with values 0, 1, 2, ... to have something non-trivial
    std::iota(proj_data2.begin(), proj_data2.end(), 0.F);
    proj_data.fill(proj_data2);
    ProjDataInMemory proj_data3(proj_data2);
    // check if equal by subtracting
    proj_data3.sapyb(1.F, proj_data2, -1.F);
    check(norm(proj_data3.begin(), proj_data3.end()) <= 0.0001F * norm(proj_data2.begin(), proj_data2.end()),
          "fill(ProjDataInMemory&");
  }
  timer.stop();
  std::cerr << "-- CPU Time " << timer.value() << '\n';

  std::cerr << "\ntest fill(ProjDataInterfile)\n";
  timer.reset();
  timer.start();
  {
    ProjDataInterfile proj_data2(proj_data.get_exam_info_sptr(),
                                 proj_data.get_proj_data_info_sptr(),
                                 "test_proj_data_fill.hs",
                                 std::ios::in | std::ios::out | std::ios::trunc);
    proj_data2.fill(value);
    for (int seg = proj_data.get_min_segment_num(); seg <= proj_data.get_max_segment_num(); ++seg)
      {
        auto viewgram = proj_data.get_empty_viewgram(1, seg, false, proj_data.get_max_tof_pos_num());
        viewgram.fill(value * seg);
        check(proj_data2.set_viewgram(viewgram) == Succeeded::yes, "test set_viewgram succeeded");
        auto sinogram = proj_data.get_empty_sinogram(1, seg, false, proj_data.get_min_tof_pos_num());
        sinogram.fill(value * 3.12 * seg);
        check(proj_data2.set_sinogram(sinogram) == Succeeded::yes, "test set_sinogram succeeded");
      }
    proj_data.fill(proj_data2);
    // check if equal by subtracting
    ProjDataInMemory proj_data3(proj_data);
    const auto proj_data_norm = norm(proj_data3.begin(), proj_data3.end());
    proj_data3.sapyb(1.F, proj_data2, -1.F);
    check(norm(proj_data3.begin(), proj_data3.end()) <= 0.0001F * proj_data_norm, "fill(ProjDataInterfile&)");
    if (!this->is_everything_ok())
      exit(1);
  }
  timer.stop();
  std::cerr << "-- CPU Time " << timer.value() << '\n';

  std::cerr << "\ntest making a copy using stir::copy_to\n";
  timer.reset();
  timer.start();
  {
    ProjDataInMemory proj_data2(proj_data.get_exam_info_sptr(), proj_data.get_proj_data_info_sptr());
    ProjData const& p = proj_data;
    copy_to(p, proj_data2.begin_all());
    check_if_equal(proj_data2.get_viewgram(0, 0).find_max(),
                   proj_data.get_viewgram(0, 0).find_max(),
                   "test 1 for templated-copy and get_viewgram(0,0)");
    check_if_equal(proj_data2.get_viewgram(1, 1).find_max(),
                   proj_data.get_viewgram(1, 1).find_max(),
                   "test 1 for templated-copy and get_viewgram(1,1)");
  }
  timer.stop();
  std::cerr << "-- CPU Time " << timer.value() << '\n';
  std::cerr << "\ntest making a copy using stir::copy_to with reference to ProjData\n";
  timer.reset();
  timer.start();
  {
    ProjDataInMemory proj_data2(proj_data.get_exam_info_sptr(), proj_data.get_proj_data_info_sptr());
    ProjData& p_ref(proj_data);
    copy_to(p_ref, proj_data2.begin_all());
    check_if_equal(proj_data2.get_viewgram(0, 0).find_max(),
                   proj_data.get_viewgram(0, 0).find_max(),
                   "test 1 for templated-copy ProjData& and get_viewgram(0,0)");
    check_if_equal(proj_data2.get_viewgram(1, 1).find_max(),
                   proj_data.get_viewgram(1, 1).find_max(),
                   "test 1 for templated-copy ProjData& and get_viewgram(1,1)");
  }
  timer.stop();
  std::cerr << "-- CPU Time " << timer.value() << '\n';

  std::cerr << "\ntest consistency of copy_to and fill_from\n";
  timer.reset();
  timer.start();
  {
    ProjDataInMemory proj_data2(proj_data.get_exam_info_sptr(), proj_data.get_proj_data_info_sptr());
    // fill with values 0, 1, 2, ... to have something non-trivial
    std::iota(proj_data2.begin(), proj_data2.end(), 0.F);
    proj_data.fill_from(proj_data2.begin());
    ProjDataInMemory proj_data3(proj_data.get_exam_info_sptr(), proj_data.get_proj_data_info_sptr());
    proj_data.copy_to(proj_data3.begin());
    // check if equal by subtracting
    proj_data3.sapyb(1.F, proj_data2, -1.F);
    check(norm(proj_data3.begin(), proj_data3.end()) <= 0.0001F * norm(proj_data2.begin(), proj_data2.end()),
          "copy_to/fill_from consistency");
  }
  timer.stop();
  std::cerr << "-- CPU Time " << timer.value() << '\n';

  std::cerr << "\ntest copy_to order\n";
  timer.reset();
  timer.start();
  {
    Array<4, float> test_array(IndexRange4D(proj_data.get_min_tof_pos_num(),
                                            proj_data.get_max_tof_pos_num(),
                                            0,
                                            proj_data.get_num_non_tof_sinograms() - 1,
                                            proj_data.get_min_view_num(),
                                            proj_data.get_max_view_num(),
                                            proj_data.get_min_tangential_pos_num(),
                                            proj_data.get_max_tangential_pos_num()));
    // copy to the array
    copy_to(proj_data, test_array.begin_all());
    for (int k = proj_data.get_min_tof_pos_num(); k <= proj_data.get_max_tof_pos_num(); ++k)
      {
        int total_ax_pos_num = 0;
        for (int segment_num : proj_data.standard_segment_sequence(*proj_data.get_proj_data_info_sptr()))
          {
            for (int ax_pos_num = proj_data.get_min_axial_pos_num(segment_num);
                 ax_pos_num <= proj_data.get_max_axial_pos_num(segment_num);
                 ++ax_pos_num, ++total_ax_pos_num)
              {
                if (!check_if_equal(proj_data.get_sinogram(ax_pos_num, segment_num, false, k),
                                    test_array[k][total_ax_pos_num],
                                    "test copy_to order"))
                  break; // get out, at least of this loop
              }
          }
      }
  }
  timer.stop();
  std::cerr << "-- CPU Time " << timer.value() << '\n';

  // find original span and max ring diff
  int span, max_ring_diff;
  bool arc_corrected;
  {
    const auto& pdi = dynamic_cast<ProjDataInfoCylindrical const&>(*proj_data.get_proj_data_info_sptr());
    max_ring_diff = pdi.get_max_ring_difference(pdi.get_max_segment_num());
    span = max_ring_diff - pdi.get_min_ring_difference(pdi.get_max_segment_num()) + 1;
    arc_corrected = !is_null_ptr(dynamic_cast<ProjDataInfoCylindricalArcCorr const*>(proj_data.get_proj_data_info_sptr().get()));
  }

  std::cerr << "\ntest fill with larger input\n";
  timer.reset();
  timer.start();
  {
    shared_ptr<ProjDataInfo> proj_data_info_sptr2(
        ProjDataInfo::construct_proj_data_info(proj_data.get_proj_data_info_sptr()->get_scanner_sptr(),
                                               span,
                                               std::max(max_ring_diff - span, 0),
                                               proj_data.get_num_views(),
                                               proj_data.get_num_tangential_poss(),
                                               arc_corrected,
                                               proj_data.get_proj_data_info_sptr()->get_tof_mash_factor()));

    // construct without filling
    ProjDataInMemory proj_data2(proj_data.get_exam_info_sptr(), proj_data_info_sptr2, false);
    proj_data2.fill(proj_data);
    check_if_equal(proj_data2.get_viewgram(0, 0).find_max(),
                   proj_data.get_viewgram(0, 0).find_max(),
                   "test 1 for copy-constructor and get_viewgram");
    check_if_equal(proj_data2.get_viewgram(1, 1).find_max(),
                   proj_data.get_viewgram(1, 1).find_max(),
                   "test 1 for copy-constructor and get_viewgram");
  }

  timer.stop();
  std::cerr << "-- CPU Time " << timer.value() << '\n';
  std::cerr << "\ntest fill with smaller input\n";
  timer.reset();
  timer.start();
  {
    shared_ptr<ProjDataInfo> proj_data_info_sptr2(
        ProjDataInfo::construct_proj_data_info(proj_data.get_proj_data_info_sptr()->get_scanner_sptr(),
                                               span,
                                               max_ring_diff + 2,
                                               proj_data.get_num_views(),
                                               proj_data.get_num_tangential_poss(),
                                               arc_corrected,
                                               proj_data.get_proj_data_info_sptr()->get_tof_mash_factor()));

    // construct without filling
    ProjDataInMemory proj_data2(proj_data.get_exam_info_sptr(), proj_data_info_sptr2, false);
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
  timer.stop();
  std::cerr << "-- CPU Time " << timer.value() << '\n';
}

void
ProjDataTests::run_tests_in_memory_only(ProjDataInMemory& proj_data)
{
  std::cerr << "\ntest set_bin_value() and get_bin_value\n";
  {
    std::vector<float> test;
    test.resize(proj_data.size_all());

    for (unsigned int i = 0; i < test.size(); i++)
      test[i] = i;

    fill_from(proj_data, test.begin(), test.end());

    Bin bin(0, proj_data.get_max_view_num() / 2, proj_data.get_max_axial_pos_num(0) / 2, 0);

    bin.set_bin_value(42);
    proj_data.set_bin_value(bin);
    check_if_equal(
        bin.get_bin_value(), proj_data.get_bin_value(bin), "ProjDataInMemory::set_bin_value/get_bin_value not consistent");
    // also check via get_viewgram
    const Viewgram<float> viewgram = proj_data.get_viewgram(bin.view_num(), bin.segment_num());
    check_if_equal(bin.get_bin_value(),
                   viewgram[bin.axial_pos_num()][bin.tangential_pos_num()],
                   "ProjDataInMemory::set_bin_value/get_viewgram not consistent");
  }
  std::cerr << "test if copy_to is consistent with iterators\n";
  {
    Array<4, float> test_array(IndexRange4D(proj_data.get_num_tof_poss(),
                                            proj_data.get_num_non_tof_sinograms(),
                                            proj_data.get_num_views(),
                                            proj_data.get_num_tangential_poss()));
    // copy to the array
    copy_to(proj_data, test_array.begin_all());

    {
      auto test_array_iter = test_array.begin_all_const();
      auto proj_data_iter = proj_data.begin_all();
      while (test_array_iter != test_array.end_all_const())
        {
          if (!check_if_equal(*proj_data_iter, *test_array_iter, "check if array iterator in correct order"))
            {
              // get out as there will be lots of other failures
              break;
            }
          ++test_array_iter;
          ++proj_data_iter;
        }
    }
  }
}

void
ProjDataTests::run_tests()
{
  std::cerr << "-------- Testing ProjData --------\n";

  shared_ptr<ExamInfo> exam_info_sptr(new ExamInfo);
  exam_info_sptr->imaging_modality = ImagingModality::PT;

  std::cerr << "\n--------------------------------non-TOF tests\n";

  {
    shared_ptr<Scanner> scanner_sptr(new Scanner(Scanner::E953));

    // the test uses a non-standard number of views at the moment.
    // Just to see if that works as well :-)
    shared_ptr<ProjDataInfo> proj_data_info_sptr(ProjDataInfo::ProjDataInfoCTI(scanner_sptr,
                                                                               /*span*/ 1,
                                                                               10,
                                                                               /*views*/ 95,
                                                                               /*tang_pos*/ 132,
                                                                               /*arc_corrected*/ true));

    // construct with filling to 0
    ProjDataInMemory proj_data_in_memory(exam_info_sptr, proj_data_info_sptr);
    {
      Sinogram<float> sinogram = proj_data_in_memory.get_sinogram(0, 0);
      check_if_equal(sinogram.find_min(), 0.F, "test constructor and get_sinogram");
    }

    run_tests_on_proj_data(proj_data_in_memory);
    run_tests_in_memory_only(proj_data_in_memory);

    std::cerr << "\n-----------------Repeating tests but now with interfile input\n";

    ProjDataInterfile(exam_info_sptr, proj_data_info_sptr, "test_proj_data.hs", std::ios::in | std::ios::out | std::ios::trunc);
    run_tests_on_proj_data(proj_data_in_memory);
  }

  std::cerr << "\n--------------------------------TOF tests\n";

  {
    shared_ptr<Scanner> scanner_sptr(new Scanner(Scanner::Discovery690));

    shared_ptr<ProjDataInfo> proj_data_info_sptr(
        ProjDataInfo::construct_proj_data_info(scanner_sptr,
                                               /*span*/ 2,
                                               5,
                                               /*views*/ scanner_sptr->get_num_detectors_per_ring() / 4,
                                               /*tang_pos*/ 22,
                                               /*arc_corrected*/ false,
                                               /* Tof_mashing */ 11));

    // construct with filling to 0
    ProjDataInMemory proj_data_in_memory(exam_info_sptr, proj_data_info_sptr);
    {
      Sinogram<float> sinogram = proj_data_in_memory.get_sinogram(0, 0);
      check_if_equal(sinogram.find_min(), 0.F, "test constructor and get_sinogram");
    }

    std::cerr << "\n----------------- Tests with ProjDataInMemory\n";
    run_tests_on_proj_data(proj_data_in_memory);
    run_tests_in_memory_only(proj_data_in_memory);

    std::cerr << "\n-----------------Repeating tests but now with interfile input\n";

    ProjDataInterfile proj_data_interfile(
        exam_info_sptr, proj_data_info_sptr, "test_proj_data.hs", std::ios::in | std::ios::out | std::ios::trunc);
    run_tests_on_proj_data(proj_data_interfile);
  }
}
END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int
main()
{
  ProjDataTests tests;
  tests.run_tests();
  return tests.main_return_value();
}
