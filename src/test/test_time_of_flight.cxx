/*
    Copyright (C) 2016, 2022, 2025, UCL
    Copyright (C) 2016, University of Hull
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0
    See STIR/LICENSE.txt for details
*/

#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/DetectionPositionPair.h"
#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#include "stir/recon_buildblock/ProjMatrixElemsForOneBin.h"
#include "stir/HighResWallClockTimer.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/Succeeded.h"
#include "stir/shared_ptr.h"
#include "stir/RunTests.h"
#include "stir/Scanner.h"
//#include "stir/stream.h"

#ifdef HAVE_CERN_ROOT
#  include "stir/listmode/CListRecordROOT.h"
#endif
#include "stir/info.h"
#include "stir/warning.h"
#include <cmath>
#include <algorithm>

START_NAMESPACE_STIR

// Helper class.
class FloatFloat
{
public:
  FloatFloat()
  {
    float1 = 0.f;
    float2 = 0.f;
  }
  float float1;
  float float2;
};

/*!
  \ingroup test
  \brief Test class for Time Of Flight
  \author Nikos Efthimiou


  The following tests are performed:

  *. Compare the ProjDataInfo of the GE Signa scanner to known values.

  *. Check if get_det_pos_pair_for_bin swaps detectors (or timing_pos) for bins with opposite timing_pos

  *. Check that the sum of the TOF LOR is the same as the non TOF.

  *. Check if the back-projection of the first and last TOF bin are symmetric for an oblique LOR

  \warning If you change the mashing factor the test_tof_proj_data_info() will fail.
  \warning The execution time strongly depends on the value of the TOF mashing factor
*/
class TOF_Tests : public RunTests
{
public:
  void run_tests() override;

private:
  void test_tof_proj_data_info_kernel();
  void test_tof_proj_data_info_det_pos();
  void test_tof_proj_data_info();
#ifdef HAVE_CERN_ROOT
  void test_CListEventROOT();
#endif
  //! This check picks a specific bin, finds the LOR and applies all the
  //! kernels of all available timing positions. Then check if the sum
  //! of the TOF bins is equal to the non-TOF LOR.
  void test_tof_kernel_application();

  //! Check if the back-projection of the first and last TOF bin are symmetric for an oblique LOR
  void test_tof_kernel_application_is_symmetric();

  shared_ptr<Scanner> test_scanner_sptr;
  shared_ptr<ProjDataInfo> test_proj_data_info_sptr;
  shared_ptr<ProjDataInfo> test_nonTOF_proj_data_info_sptr;

  shared_ptr<DiscretisedDensity<3, float>> test_discretised_density_sptr;
  shared_ptr<ProjMatrixByBin> test_proj_matrix_sptr;
  shared_ptr<ProjMatrixByBin> test_nonTOF_proj_matrix_sptr;
};

void
TOF_Tests::run_tests()
{
  // New Scanner
  test_scanner_sptr.reset(new Scanner(Scanner::PETMR_Signa));

  // New Proj_Data_Info
  const int test_tof_mashing_factor = 39; // to have 9 TOF bins (381/39=9)
  test_proj_data_info_sptr.reset(ProjDataInfo::ProjDataInfoCTI(test_scanner_sptr,
                                                               1,
                                                               test_scanner_sptr->get_num_rings() - 1,
                                                               test_scanner_sptr->get_num_detectors_per_ring() / 2,
                                                               test_scanner_sptr->get_max_num_non_arccorrected_bins(),
                                                               /* arc_correction*/ false));
  test_proj_data_info_sptr->set_tof_mash_factor(test_tof_mashing_factor);

  test_nonTOF_proj_data_info_sptr.reset(ProjDataInfo::ProjDataInfoCTI(test_scanner_sptr,
                                                                      1,
                                                                      test_scanner_sptr->get_num_rings() - 1,
                                                                      test_scanner_sptr->get_num_detectors_per_ring() / 2,
                                                                      test_scanner_sptr->get_max_num_non_arccorrected_bins(),
                                                                      /* arc_correction*/ false));
  test_nonTOF_proj_data_info_sptr->set_tof_mash_factor(0);

  test_tof_proj_data_info();
  //    test_tof_geometry_1();

#ifdef HAVE_CERN_ROOT
  test_CListEventROOT();
#endif

  // New Discretised Density
  test_discretised_density_sptr.reset(new VoxelsOnCartesianGrid<float>(
      *test_proj_data_info_sptr, 1.f, CartesianCoordinate3D<float>(0.f, 0.f, 0.f), CartesianCoordinate3D<int>(-1, -1, -1)));
  // New ProjMatrix
  test_proj_matrix_sptr.reset(new ProjMatrixByBinUsingRayTracing());
  dynamic_cast<ProjMatrixByBinUsingRayTracing*>(test_proj_matrix_sptr.get())->set_num_tangential_LORs(1);
  dynamic_cast<ProjMatrixByBinUsingRayTracing*>(test_proj_matrix_sptr.get())
      ->set_up(test_proj_data_info_sptr, test_discretised_density_sptr);

  test_nonTOF_proj_matrix_sptr.reset(new ProjMatrixByBinUsingRayTracing());
  dynamic_cast<ProjMatrixByBinUsingRayTracing*>(test_nonTOF_proj_matrix_sptr.get())->set_num_tangential_LORs(1);
  dynamic_cast<ProjMatrixByBinUsingRayTracing*>(test_nonTOF_proj_matrix_sptr.get())
      ->set_up(test_nonTOF_proj_data_info_sptr, test_discretised_density_sptr);

  test_tof_kernel_application();
  test_tof_kernel_application_is_symmetric();
}

void
TOF_Tests::test_tof_proj_data_info_kernel()
{
  const int correct_tof_mashing_factor = 39;
  const int num_timing_positions = test_scanner_sptr->get_max_num_timing_poss() / correct_tof_mashing_factor;
  const float correct_width_of_tof_bin
      = test_scanner_sptr->get_size_of_timing_pos() * test_proj_data_info_sptr->get_tof_mash_factor() * 0.299792458f / 2;
  std::vector<float> correct_timing_locations(num_timing_positions);
  for (int i = 0; i < num_timing_positions; ++i)
    {
      correct_timing_locations[i] = (i - (num_timing_positions - 1) / 2.F) * correct_width_of_tof_bin;
    }

  check_if_equal(correct_tof_mashing_factor, test_proj_data_info_sptr->get_tof_mash_factor(), "Different TOF mashing factor.");

  check_if_equal(num_timing_positions, test_proj_data_info_sptr->get_num_tof_poss(), "Different number of timing positions.");

  for (int timing_pos_num = test_proj_data_info_sptr->get_min_tof_pos_num(), counter = 0;
       timing_pos_num <= test_proj_data_info_sptr->get_max_tof_pos_num();
       ++timing_pos_num, counter++)
    {
      Bin bin(0, 0, 0, 0, timing_pos_num, 1.f);

      check_if_equal(static_cast<double>(correct_width_of_tof_bin),
                     static_cast<double>(test_proj_data_info_sptr->get_sampling_in_k(bin)),
                     "Error in get_sampling_in_k()");
      check_if_equal(static_cast<double>(correct_timing_locations[counter]),
                     static_cast<double>(test_proj_data_info_sptr->get_k(bin)),
                     "Error in get_sampling_in_k()");
    }

  float total_width = test_proj_data_info_sptr->get_k(Bin(0, 0, 0, 0, test_proj_data_info_sptr->get_max_tof_pos_num(), 1.f))
                      - test_proj_data_info_sptr->get_k(Bin(0, 0, 0, 0, test_proj_data_info_sptr->get_min_tof_pos_num(), 1.f))
                      + test_proj_data_info_sptr->get_sampling_in_k(Bin(0, 0, 0, 0, 0, 1.f));

  set_tolerance(static_cast<double>(0.005));
  check_if_equal(static_cast<double>(total_width),
                 static_cast<double>(test_proj_data_info_sptr->get_scanner_ptr()->get_coincidence_window_width_in_mm()),
                 "Coincidence widths don't match.");
}

void
TOF_Tests::test_tof_proj_data_info_det_pos()
{

  auto pdi_ptr = dynamic_cast<ProjDataInfoCylindricalNoArcCorr const*>(test_proj_data_info_sptr.get());

  Bin b1(1, 2, 3, 4, 5);
  Bin b2 = b1;
  b2.timing_pos_num() = -b1.timing_pos_num();

  DetectionPositionPair<> dp1, dp2;
  pdi_ptr->get_det_pos_pair_for_bin(dp1, b1);
  pdi_ptr->get_det_pos_pair_for_bin(dp2, b2);

  check((dp1.timing_pos() == dp2.timing_pos() && dp1.pos1() == dp2.pos2() && dp1.pos2() == dp2.pos1())
            || (static_cast<int>(dp1.timing_pos()) == -static_cast<int>(dp2.timing_pos()) && dp1.pos1() == dp2.pos1()
                && dp1.pos2() == dp2.pos2()),
        "get_det_pos_for_bin with bins of opposite timing_pos");
}

void
TOF_Tests::test_tof_proj_data_info()
{
  test_tof_proj_data_info_kernel();
  test_tof_proj_data_info_det_pos();
}

#ifdef HAVE_CERN_ROOT
void
TOF_Tests::test_CListEventROOT()
{
  std::cerr << "CListEventROOT tests\n";
  const auto old_tol = this->get_tolerance();
  // set tolerance to ~1mm. It has to be surprisingly large at the moment. Problem in the LOR functions? (TODO)
  this->set_tolerance(3.F);

  test_proj_data_info_sptr->set_tof_mash_factor(1);

  const int ring1 = 1, ring2 = 4, crystal1 = 0, crystal2 = 25;
  const float delta_time = 800.F;

  CListEventROOT event(test_proj_data_info_sptr);
  event.init_from_data(ring1, ring2, crystal1, crystal2, delta_time);
  Bin bin;
  // this doesn't set time_frame, so force that to 1 for later comparisons
  bin.time_frame_num() = 1;

  event.get_bin(bin, *test_proj_data_info_sptr);
  check(bin.timing_pos_num() != 0, "test CListEventROOT non-zero TOF bin");

  DetectionPositionPair<> det_pos;
  event.get_detection_position(det_pos);
  LORAs2Points<float> lor_2pts(event.get_LOR());
  LORInAxialAndNoArcCorrSinogramCoordinates<float> lor_sc;
  test_proj_data_info_sptr->get_LOR(lor_sc, bin);
  LORAs2Points<float> test_lor(lor_sc);
  check_if_equal(lor_2pts.p1(), test_lor.p1(), "CListEventROOT::get_LOR and ProjDataInfo::get_LOR consistency check 1");
  check_if_equal(lor_2pts.p2(), test_lor.p2(), "CListEventROOT::get_LOR and ProjDataInfo::get_LOR consistency check 2");

  event.init_from_data(ring2, ring1, crystal2, crystal1, -delta_time);
  {
    Bin bin_swapped;
    bin_swapped.time_frame_num() = 1;
    event.get_bin(bin_swapped, *test_proj_data_info_sptr);
    check_if_equal(bin_swapped, bin, "CListEventROOT: get_bin with swapped detectors");
    {
      DetectionPositionPair<> det_pos_swapped;
      event.get_detection_position(det_pos_swapped);
      if (det_pos_swapped.timing_pos() == det_pos.timing_pos())
        {
          check_if_equal(det_pos_swapped.pos1(),
                         det_pos.pos1(),
                         "CListEventROOT: get_detection_position with swapped detectors: equal timing_pos, but different pos1");
          check_if_equal(det_pos_swapped.pos2(),
                         det_pos.pos2(),
                         "CListEventROOT: get_detection_position with swapped detectors: equal timing_pos, but different pos2");
        }
      else if (det_pos_swapped.timing_pos() == -det_pos.timing_pos())
        {
          check_if_equal(
              det_pos_swapped.pos2(),
              det_pos.pos1(),
              "CListEventROOT: get_detection_position with swapped detectors: opposite timing_pos, but different pos1/2");
          check_if_equal(
              det_pos_swapped.pos1(),
              det_pos.pos2(),
              "CListEventROOT: get_detection_position with swapped detectors: opposite timing_pos, but different pos1/2");
        }
      else
        {
          check_if_equal(std::abs(det_pos_swapped.timing_pos()),
                         std::abs(det_pos.timing_pos()),
                         "CListEventROOT: get_detection_position with swapped detectors: wrong timing_pos");
        }
    }

    LORAs2Points<float> lor_2pts_swapped(event.get_LOR());
    LORInAxialAndNoArcCorrSinogramCoordinates<float> lor_sc_swapped;
    test_proj_data_info_sptr->get_LOR(lor_sc_swapped, bin_swapped);
    LORAs2Points<float> test_lor_swapped(lor_sc);
    check_if_equal(
        lor_2pts_swapped.p1(), test_lor_swapped.p1(), "CListEventROOT::get_LOR and ProjDataInfo::get_LOR consistency check 3");
    check_if_equal(
        lor_2pts_swapped.p2(), test_lor_swapped.p2(), "CListEventROOT::get_LOR and ProjDataInfo::get_LOR consistency check 4");

    // now check if equal
    check_if_equal(bin, bin_swapped, "CListEventROOT:get_bin for reordered detectors");
    check_if_equal(lor_2pts_swapped.p1(), lor_2pts.p1(), "CListEventROOT::get_LOR and ProjDataInfo::get_LOR consistency check 5");
    check_if_equal(lor_2pts_swapped.p2(), lor_2pts.p2(), "CListEventROOT::get_LOR and ProjDataInfo::get_LOR consistency check 6");
  }
  // repeat with swapped detectors
  this->set_tolerance(old_tol);
}
#endif

void
TOF_Tests::test_tof_kernel_application()
{
  int seg_num = 3;
  int view_num = 2;
  int axial_num = 1;
  int tang_num = 4;

  ProjMatrixElemsForOneBin proj_matrix_row;
  ProjMatrixElemsForOneBin sum_tof_proj_matrix_row;

  HighResWallClockTimer t;
  std::vector<double> times_of_tofing;

  auto proj_data_ptr = test_proj_data_info_sptr.get();

  LORInAxialAndNoArcCorrSinogramCoordinates<float> lor;

  Bin this_bin(seg_num, view_num, axial_num, tang_num, 1.f);

  t.reset();
  t.start();
  test_nonTOF_proj_matrix_sptr->get_proj_matrix_elems_for_one_bin(proj_matrix_row, this_bin);
  t.stop();

  std::cerr << "Execution time for nonTOF: " << t.value() << std::endl;
  proj_data_ptr->get_LOR(lor, this_bin);
  LORAs2Points<float> lor2(lor);

  for (int timing_pos_num = test_proj_data_info_sptr->get_min_tof_pos_num();
       timing_pos_num <= test_proj_data_info_sptr->get_max_tof_pos_num();
       ++timing_pos_num)
    {
      ProjMatrixElemsForOneBin new_proj_matrix_row;
      Bin bin(seg_num, view_num, axial_num, tang_num, timing_pos_num, 1.f);

      t.reset();
      t.start();
      test_proj_matrix_sptr->get_proj_matrix_elems_for_one_bin(new_proj_matrix_row, bin);
      t.stop();
      times_of_tofing.push_back(t.value());

      if (sum_tof_proj_matrix_row.size() > 0)
        sum_tof_proj_matrix_row.merge(new_proj_matrix_row);
      else
        sum_tof_proj_matrix_row = new_proj_matrix_row;
    }

  // Get value of nonTOF LOR, for central voxels only
  float nonTOF_val = 0.0;
  float TOF_val = 0.0;

  {
    ProjMatrixElemsForOneBin::iterator element_ptr = proj_matrix_row.begin();
    while (element_ptr != proj_matrix_row.end())
      {
        if (element_ptr->get_value() > nonTOF_val)
          nonTOF_val = element_ptr->get_value();
        ++element_ptr;
      }
  }

  // Get value of TOF LOR, for central voxels only

  {
    ProjMatrixElemsForOneBin::iterator element_ptr = sum_tof_proj_matrix_row.begin();
    while (element_ptr != sum_tof_proj_matrix_row.end())
      {
        if (element_ptr->get_value() > TOF_val)
          TOF_val = element_ptr->get_value();
        ++element_ptr;
      }
  }

  check_if_equal(
      static_cast<double>(nonTOF_val), static_cast<double>(TOF_val), "Sum over nonTOF LOR does not match sum over TOF LOR.");

  // report timings
  {
    double mean = std::accumulate(times_of_tofing.begin(), times_of_tofing.end(), 0.) / (times_of_tofing.size());

    double s = 0.0;
    for (unsigned i = 0; i < times_of_tofing.size(); i++)
      s += square(times_of_tofing.at(i) - mean) / (times_of_tofing.size() - 1);

    s = std::sqrt(s);
    std::cerr << "Execution  time  for TOF: " << mean << " ±" << s;
  }

  std::cerr << std::endl;
}

/*!
  Check the matrix rows for the first and last TOF bin (for an oblique LOR). The
  detection probabilities should be symmetric w.r.t. eachother.

  Ideally, we'd also check the voxel indices, but that is not implemented yet.
*/
void
TOF_Tests::test_tof_kernel_application_is_symmetric()
{
  int seg_num = test_proj_data_info_sptr->get_max_segment_num();
  int view_num = 0;
  int axial_num
      = (test_proj_data_info_sptr->get_min_axial_pos_num(seg_num) + test_proj_data_info_sptr->get_max_axial_pos_num(seg_num)) / 2;
  int tang_num = 0;

  ProjMatrixElemsForOneBin proj_matrix_row;
  Bin this_bin(seg_num, view_num, axial_num, tang_num, test_proj_data_info_sptr->get_max_tof_pos_num(), 1.f);
  test_proj_matrix_sptr->get_proj_matrix_elems_for_one_bin(proj_matrix_row, this_bin);

  ProjMatrixElemsForOneBin proj_matrix_row2;
  this_bin.timing_pos_num() = test_proj_data_info_sptr->get_min_tof_pos_num();
  test_proj_matrix_sptr->get_proj_matrix_elems_for_one_bin(proj_matrix_row2, this_bin);

  // check if symmetric
  {
    auto riter = proj_matrix_row.rbegin();
    auto iter2 = proj_matrix_row2.begin();
    while (riter != proj_matrix_row.rend())
      {
        // std::cerr << riter->get_coords() << "," << riter->get_value() << iter2->get_coords() << iter2->get_value() << "\n";
        check_if_equal(riter->get_value(), iter2->get_value(), "check symmetry in TOF");
        ++iter2;
        ++riter;
      }
  }
}

END_NAMESPACE_STIR

int
main()
{
  USING_NAMESPACE_STIR
  TOF_Tests tests;
  tests.run_tests();
  return tests.main_return_value();
}
