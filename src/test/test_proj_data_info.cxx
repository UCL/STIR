//
//
/*!

  \file
  \ingroup test

  \brief Test program for stir::ProjDataInfo hierarchy

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author Palak Wadhwa
  \author Daniel Deidda
  \author PARAPET project

*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2011, Hammersmith Imanet Ltd
    Copyright (C) 2018, 2021, 2022, University College London
    Copyright (C) 2018, University of Leeds
    Copyright (C) 2021, National Physical Laboratory
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/

#include "stir/ProjDataInfoCylindricalArcCorr.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/LORCoordinates.h"
#include "stir/ProjDataInfo.h"
#include "stir/ProjDataInfoBlocksOnCylindricalNoArcCorr.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/RunTests.h"
#include "stir/Scanner.h"
#include "stir/Bin.h"
#include "stir/LORCoordinates.h"
#include "stir/round.h"
#include "stir/num_threads.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <math.h>
#include "stir/CPUTimer.h"

using std::cerr;
using std::setw;
using std::endl;
using std::min;
using std::max;
using std::size_t;

//#define STIR_TOF_DEBUG 1

START_NAMESPACE_STIR

static inline int
intabs(const int x)
{
  return x >= 0 ? x : -x;
}

// prints a michelogram to the screen
#if 1
// TODO move somewhere else
void
michelogram(const ProjDataInfoCylindrical& proj_data_info)
{
  cerr << '{';
  for (int ring1 = 0; ring1 < proj_data_info.get_scanner_ptr()->get_num_rings(); ++ring1)
    {
      cerr << '{';
      for (int ring2 = 0; ring2 < proj_data_info.get_scanner_ptr()->get_num_rings(); ++ring2)
        {
          int segment_num = 0;
          int ax_pos_num = 0;
          if (proj_data_info.get_segment_axial_pos_num_for_ring_pair(segment_num, ax_pos_num, ring1, ring2) == Succeeded::yes)
            cerr << '{' << setw(3) << segment_num << ',' << setw(2) << ax_pos_num << "}";
          else
            cerr << "{}      ";
          if (ring2 != proj_data_info.get_scanner_ptr()->get_num_rings() - 1)
            cerr << ',';
        }
      cerr << '}';
      if (ring1 != proj_data_info.get_scanner_ptr()->get_num_rings() - 1)
        cerr << ',';
      cerr << endl;
    }
  cerr << '}' << endl;
}
#endif

/*!
  \ingroup test
  \brief Test class for ProjDataInfo
*/
class ProjDataInfoTests : public RunTests
{
protected:
  void test_generic_proj_data_info(ProjDataInfo& proj_data_info);

  template <class TProjDataInfo>
  shared_ptr<TProjDataInfo> set_blocks_projdata_info(shared_ptr<Scanner> scanner_sptr);
  void run_coordinate_test();
  void run_coordinate_test_for_realistic_scanner();
  void run_Blocks_DOI_test();
  void run_lor_get_s_test();
};

/*! The following is a function to allow a projdata_info blocksONCylindrical to be created from the scanner.
 */
template <class TProjDataInfo>
shared_ptr<TProjDataInfo>
ProjDataInfoTests::set_blocks_projdata_info(shared_ptr<Scanner> scanner_sptr)
{
  VectorWithOffset<int> num_axial_pos_per_segment(scanner_sptr->get_num_rings() * 2 - 1);
  VectorWithOffset<int> min_ring_diff_v(scanner_sptr->get_num_rings() * 2 - 1);
  VectorWithOffset<int> max_ring_diff_v(scanner_sptr->get_num_rings() * 2 - 1);
  for (int i = 0; i < 2 * scanner_sptr->get_num_rings() - 1; i++)
    {
      min_ring_diff_v[i] = -scanner_sptr->get_num_rings() + 1 + i;
      max_ring_diff_v[i] = -scanner_sptr->get_num_rings() + 1 + i;
      if (i < scanner_sptr->get_num_rings())
        num_axial_pos_per_segment[i] = i + 1;
      else
        num_axial_pos_per_segment[i] = 2 * scanner_sptr->get_num_rings() - i - 1;
    }

  auto proj_data_info_blocks_sptr = std::make_shared<TProjDataInfo>(scanner_sptr,
                                                                    num_axial_pos_per_segment,
                                                                    min_ring_diff_v,
                                                                    max_ring_diff_v,
                                                                    scanner_sptr->get_max_num_views(),
                                                                    scanner_sptr->get_max_num_non_arccorrected_bins());

  return proj_data_info_blocks_sptr;
}

void
ProjDataInfoTests::test_generic_proj_data_info(ProjDataInfo& proj_data_info)
{
  cerr << "\tTests on get_min/max_num\n";
  check_if_equal(proj_data_info.get_max_tangential_pos_num() - proj_data_info.get_min_tangential_pos_num() + 1,
                 proj_data_info.get_num_tangential_poss(),
                 "basic check on min/max/num_tangential_pos_num");
  check(abs(proj_data_info.get_max_tangential_pos_num() + proj_data_info.get_min_tangential_pos_num()) <= 1,
        "check on min/max_tangential_pos_num being (almost) centred");
  check_if_equal(proj_data_info.get_max_tof_pos_num() - proj_data_info.get_min_tof_pos_num() + 1,
                 proj_data_info.get_num_tof_poss(),
                 "basic check on min/max/num_tof_pos_num");
  check_if_equal(proj_data_info.get_max_tof_pos_num() + proj_data_info.get_min_tof_pos_num(),
                 0,
                 "check on min/max_tof_pos_num being (almost) centred");
  check_if_equal(proj_data_info.get_max_view_num() - proj_data_info.get_min_view_num() + 1,
                 proj_data_info.get_num_views(),
                 "basic check on min/max/num_view_num");
  check_if_equal(proj_data_info.get_max_segment_num() - proj_data_info.get_min_segment_num() + 1,
                 proj_data_info.get_num_segments(),
                 "basic check on min/max/num_segment_num");
  // not strictly necessary in most of the code, but most likely required in some of it
  check_if_equal(proj_data_info.get_max_segment_num() + proj_data_info.get_min_segment_num(),
                 0,
                 "check on min/max_segment_num being centred");
  check_if_equal(proj_data_info.get_max_axial_pos_num(0) - proj_data_info.get_min_axial_pos_num(0) + 1,
                 proj_data_info.get_num_axial_poss(0),
                 "basic check on min/max/num_axial_pos_num");

  cerr << "\tTests on get_LOR/get_bin\n";
  int max_diff_segment_num = 0;
  int max_diff_view_num = 0;
  int max_diff_axial_pos_num = 0;
  int max_diff_tangential_pos_num = 0;
  int max_diff_timing_pos_num = 0;
#ifdef STIR_OPENMP
#  pragma omp parallel for schedule(dynamic)
#endif
  for (int segment_num = proj_data_info.get_min_segment_num(); segment_num <= proj_data_info.get_max_segment_num(); ++segment_num)
    {
      for (int view_num = proj_data_info.get_min_view_num(); view_num <= proj_data_info.get_max_view_num(); view_num += 1)
        {
          // loop over axial_positions. Avoid using first and last positions, as
          // if there is axial compression, the central LOR of a bin might actually not
          // fall within the scanner. In this case, the get_bin(get_LOR(org_bin)) code
          // will return an out-of-range bin (i.e. value<0).
          int axial_pos_num_margin = 0;
          const ProjDataInfoCylindrical* const proj_data_info_cyl_ptr
              = dynamic_cast<const ProjDataInfoCylindrical* const>(&proj_data_info);
          if (proj_data_info_cyl_ptr != 0)
            {
              axial_pos_num_margin = std::max(round(ceil(proj_data_info_cyl_ptr->get_average_ring_difference(segment_num)
                                                         - proj_data_info_cyl_ptr->get_min_ring_difference(segment_num))),
                                              round(ceil(proj_data_info_cyl_ptr->get_max_ring_difference(segment_num)
                                                         - proj_data_info_cyl_ptr->get_average_ring_difference(segment_num))));
            }
          for (int axial_pos_num = proj_data_info.get_min_axial_pos_num(segment_num) + axial_pos_num_margin;
               axial_pos_num <= proj_data_info.get_max_axial_pos_num(segment_num) - axial_pos_num_margin;
               axial_pos_num += 3)
            {
              for (int tangential_pos_num = proj_data_info.get_min_tangential_pos_num() + 1;
                   tangential_pos_num <= proj_data_info.get_max_tangential_pos_num() - 1;
                   tangential_pos_num += 1)
                {
                  for (int timing_pos_num = proj_data_info.get_min_tof_pos_num();
                       timing_pos_num <= proj_data_info.get_max_tof_pos_num();
                       timing_pos_num += std::max(1,
                                                  (proj_data_info.get_max_tof_pos_num() - proj_data_info.get_min_tof_pos_num())
                                                      / 2)) // take 3 or 1 steps, always going through 0
                    {
                      const Bin org_bin(segment_num, view_num, axial_pos_num, tangential_pos_num, timing_pos_num, /* value*/ 1.f);
                      const double delta_time = proj_data_info.get_tof_delta_time(org_bin);
                      LORInAxialAndNoArcCorrSinogramCoordinates<float> lor;
                      proj_data_info.get_LOR(lor, org_bin);

                      {
                        const Bin new_bin = proj_data_info.get_bin(lor, delta_time);
#if 1
                        // the differences need to also consider wrap-around in views, which would flip tangential pos and segment
                        // and TOF bin
                        const int diff_segment_num
                            = intabs(org_bin.view_num() - new_bin.view_num())
                                      < proj_data_info.get_num_views() - intabs(org_bin.view_num() - new_bin.view_num())
                                  ? intabs(org_bin.segment_num() - new_bin.segment_num())
                                  : intabs(org_bin.segment_num() + new_bin.segment_num());
                        const int diff_view_num
                            = min(intabs(org_bin.view_num() - new_bin.view_num()),
                                  proj_data_info.get_num_views() - intabs(org_bin.view_num() - new_bin.view_num()));
                        const int diff_axial_pos_num = intabs(org_bin.axial_pos_num() - new_bin.axial_pos_num());
                        const int diff_tangential_pos_num
                            = intabs(org_bin.view_num() - new_bin.view_num())
                                      < proj_data_info.get_num_views() - intabs(org_bin.view_num() - new_bin.view_num())
                                  ? intabs(org_bin.tangential_pos_num() - new_bin.tangential_pos_num())
                                  : intabs(org_bin.tangential_pos_num() + new_bin.tangential_pos_num());
                        const int diff_timing_pos_num
                            = intabs(org_bin.view_num() - new_bin.view_num())
                                      < proj_data_info.get_num_views() - intabs(org_bin.view_num() - new_bin.view_num())
                                  ? intabs(org_bin.timing_pos_num() - new_bin.timing_pos_num())
                                  : intabs(org_bin.timing_pos_num() + new_bin.timing_pos_num());
                        if (new_bin.get_bin_value() > 0)
                          {
                            if (diff_segment_num > max_diff_segment_num)
                              max_diff_segment_num = diff_segment_num;
                            if (diff_view_num > max_diff_view_num)
                              max_diff_view_num = diff_view_num;
                            if (diff_axial_pos_num > max_diff_axial_pos_num)
                              max_diff_axial_pos_num = diff_axial_pos_num;
                            if (diff_tangential_pos_num > max_diff_tangential_pos_num)
                              max_diff_tangential_pos_num = diff_tangential_pos_num;
                            if (diff_timing_pos_num > max_diff_timing_pos_num)
                              max_diff_timing_pos_num = diff_timing_pos_num;
                          }
#  ifdef STIR_OPENMP
                          // add a pragma to avoid cerr output being jumbled up if there are any errors
#    pragma omp critical(TESTPROJDATAINFO)
#  endif
                        if (!check(org_bin.get_bin_value() == new_bin.get_bin_value(), "round-trip get_LOR then get_bin: value")
                            || !check(diff_segment_num <= 0, "round-trip get_LOR then get_bin: segment")
                            || !check(diff_view_num <= 1, "round-trip get_LOR then get_bin: view")
                            || !check(diff_axial_pos_num <= 1, "round-trip get_LOR then get_bin: axial_pos")
                            || !check(diff_tangential_pos_num <= 1, "round-trip get_LOR then get_bin: tangential_pos")
                            || !check(diff_timing_pos_num == 0, "round-trip get_LOR then get_bin: timing_pos"))

#else
                        if (!check(org_bin == new_bin, "round-trip get_LOR then get_bin"))
#endif
                          {
                            cerr << "\tProblem at    segment = " << org_bin.segment_num() << ", axial pos "
                                 << org_bin.axial_pos_num() << ", view = " << org_bin.view_num()
                                 << ", tangential_pos = " << org_bin.tangential_pos_num()
                                 << ", timing_pos = " << org_bin.timing_pos_num() << "\n";
                            if (new_bin.get_bin_value() > 0)
                              cerr << "\tround-trip to segment = " << new_bin.segment_num() << ", axial pos "
                                   << new_bin.axial_pos_num() << ", view = " << new_bin.view_num()
                                   << ", tangential_pos = " << new_bin.tangential_pos_num()
                                   << ", timing_pos = " << new_bin.timing_pos_num() << '\n';
                          }
                      }
                      // repeat test but with different type of LOR
                      {
                        LORAs2Points<float> lor_as_points;
                        lor.get_intersections_with_cylinder(lor_as_points, lor.radius());
#if STIR_TOF_DEBUG > 1
                        std::cerr << "    z1=" << lor_as_points.p1().z() << ", y1=" << lor_as_points.p1().y()
                                  << ", x1=" << lor_as_points.p1().x() << "\n    z2=" << lor_as_points.p2().z()
                                  << ", y2=" << lor_as_points.p2().y() << ", x2=" << lor_as_points.p2().x() << std::endl;
#endif
                        const Bin new_bin = proj_data_info.get_bin(lor_as_points, proj_data_info.get_tof_delta_time(org_bin));
#if 1

                        // the differences need to also consider wrap-around in views, which would flip tangential pos and segment
                        const int diff_segment_num
                            = intabs(org_bin.view_num() - new_bin.view_num())
                                      < proj_data_info.get_num_views() - intabs(org_bin.view_num() - new_bin.view_num())
                                  ? intabs(org_bin.segment_num() - new_bin.segment_num())
                                  : intabs(org_bin.segment_num() + new_bin.segment_num());
                        const int diff_view_num
                            = min(intabs(org_bin.view_num() - new_bin.view_num()),
                                  proj_data_info.get_num_views() - intabs(org_bin.view_num() - new_bin.view_num()));
                        const int diff_axial_pos_num = intabs(org_bin.axial_pos_num() - new_bin.axial_pos_num());
                        const int diff_tangential_pos_num
                            = intabs(org_bin.view_num() - new_bin.view_num())
                                      < proj_data_info.get_num_views() - intabs(org_bin.view_num() - new_bin.view_num())
                                  ? intabs(org_bin.tangential_pos_num() - new_bin.tangential_pos_num())
                                  : intabs(org_bin.tangential_pos_num() + new_bin.tangential_pos_num());
                        const int diff_timing_pos_num
                            = intabs(org_bin.view_num() - new_bin.view_num())
                                      < proj_data_info.get_num_views() - intabs(org_bin.view_num() - new_bin.view_num())
                                  ? intabs(org_bin.timing_pos_num() - new_bin.timing_pos_num())
                                  : intabs(org_bin.timing_pos_num() + new_bin.timing_pos_num());
                        if (new_bin.get_bin_value() > 0)
                          {
                            if (diff_segment_num > max_diff_segment_num)
                              max_diff_segment_num = diff_segment_num;
                            if (diff_view_num > max_diff_view_num)
                              max_diff_view_num = diff_view_num;
                            if (diff_axial_pos_num > max_diff_axial_pos_num)
                              max_diff_axial_pos_num = diff_axial_pos_num;
                            if (diff_tangential_pos_num > max_diff_tangential_pos_num)
                              max_diff_tangential_pos_num = diff_tangential_pos_num;
                            if (diff_timing_pos_num > max_diff_timing_pos_num)
                              max_diff_timing_pos_num = diff_timing_pos_num;
                          }
                        if (!check(org_bin.get_bin_value() == new_bin.get_bin_value(),
                                   "round-trip get_LOR then get_bin (LORAs2Points): value")
                            || !check(diff_segment_num <= 0, "round-trip get_LOR then get_bin (LORAs2Points): segment")
                            || !check(diff_view_num <= 1, "round-trip get_LOR then get_bin (LORAs2Points): view")
                            || !check(diff_axial_pos_num <= 1, "round-trip get_LOR then get_bin (LORAs2Points): axial_pos")
                            || !check(diff_tangential_pos_num <= 1,
                                      "round-trip get_LOR then get_bin (LORAs2Points): tangential_pos")
                            || !check(diff_timing_pos_num == 0, "round-trip get_LOR then get_bin: timing_pos"))

#else
                        if (!check(org_bin == new_bin, "round-trip get_LOR then get_bin"))
#endif
                          {
                            cerr << "\tProblem at    segment = " << org_bin.segment_num() << ", axial pos "
                                 << org_bin.axial_pos_num() << ", view = " << org_bin.view_num()
                                 << ", tangential_pos_num = " << org_bin.tangential_pos_num()
                                 << ", timing_pos = " << org_bin.timing_pos_num() << "\n";
                            if (new_bin.get_bin_value() > 0)
                              cerr << "\tround-trip to segment = " << new_bin.segment_num() << ", axial pos "
                                   << new_bin.axial_pos_num() << ", view = " << new_bin.view_num()
                                   << ", tangential_pos_num = " << new_bin.tangential_pos_num()
                                   << ", timing_pos = " << new_bin.timing_pos_num() << '\n';
                          }
                      }
                    }
                }
            }
        }
    }
  cerr << "Max Deviation:  segment = " << max_diff_segment_num << ", axial pos " << max_diff_axial_pos_num
       << ", view = " << max_diff_view_num << ", tangential_pos_num = " << max_diff_tangential_pos_num
       << ", timing_pos_num = " << max_diff_timing_pos_num << "\n";

  // test on reduce_segment_range and operator>=
  {
    shared_ptr<ProjDataInfo> smaller(proj_data_info.clone());
    check(proj_data_info >= *smaller, "check on operator>= and equal objects");
    smaller->set_min_tangential_pos_num(0);
    check(proj_data_info >= *smaller, "check on tangential_pos and operator>=");
    smaller->set_min_axial_pos_num(4, 0);
    check(proj_data_info >= *smaller, "check on axial_pos and operator>=");
    // if (proj_data_info.is_tof_data())
    //   {
    //      smaller->set_min_timing_pos_num(0);
    //      check(proj_data_info >= *smaller, "check on timing_pos and operator>=");
    //   }
    smaller->reduce_segment_range(0, 0);
    check(proj_data_info >= *smaller, "check on reduce_segment_range and operator>=");
    // make one range larger, so should now fail
    smaller->set_min_tangential_pos_num(proj_data_info.get_min_tangential_pos_num() - 4);
    check(!(proj_data_info >= *smaller), "check on mixed case with tangential_pos_num and operator>=");
    // reset and do the same for axial pos
    smaller->set_min_tangential_pos_num(proj_data_info.get_min_tangential_pos_num() + 4);
    check(proj_data_info >= *smaller, "check on reduced segments and tangential_pos and operator>=");
    smaller->set_max_axial_pos_num(proj_data_info.get_max_axial_pos_num(0) + 4, 0);
    check(!(proj_data_info >= *smaller), "check on mixed case with axial_pos_num and operator>=");
  }
}

/*!
  \ingroup test
  \brief Test class for ProjDataInfoCylindrical
*/
class ProjDataInfoCylindricalTests : public ProjDataInfoTests
{
protected:
  void test_cylindrical_proj_data_info(ProjDataInfoCylindrical& proj_data_info);
};

void
ProjDataInfoCylindricalTests::test_cylindrical_proj_data_info(ProjDataInfoCylindrical& proj_data_info)
{
  cerr << "\tTesting consistency between different implementations of geometric info\n";
  {
    const Bin bin(proj_data_info.get_max_segment_num(),
                  1,
                  proj_data_info.get_max_axial_pos_num(proj_data_info.get_max_segment_num()) / 2,
                  1);
    check_if_equal(proj_data_info.get_sampling_in_m(bin),
                   proj_data_info.ProjDataInfo::get_sampling_in_m(bin),
                   "test consistency get_sampling_in_m");
    check_if_equal(proj_data_info.get_sampling_in_t(bin),
                   proj_data_info.ProjDataInfo::get_sampling_in_t(bin),
                   "test consistency get_sampling_in_t");
#if 0
    // ProjDataInfo has no default implementation for get_tantheta
    // I just leave the code here to make this explicit
    check_if_equal(proj_data_info.get_tantheta(bin),
		   proj_data_info.ProjDataInfo::get_tantheta(bin),
		   "test consistency get_tantheta");
#endif
    check_if_equal(
        proj_data_info.get_costheta(bin), proj_data_info.ProjDataInfo::get_costheta(bin), "test consistency get_costheta");

    check_if_equal(proj_data_info.get_costheta(bin),
                   cos(atan(proj_data_info.get_tantheta(bin))),
                   "cross check get_costheta and get_tantheta");

    // try the same with a non-standard ring spacing
    const float old_ring_spacing = proj_data_info.get_ring_spacing();
    proj_data_info.set_ring_spacing(2.1F);

    check_if_equal(proj_data_info.get_sampling_in_m(bin),
                   proj_data_info.ProjDataInfo::get_sampling_in_m(bin),
                   "test consistency get_sampling_in_m");
    check_if_equal(proj_data_info.get_sampling_in_t(bin),
                   proj_data_info.ProjDataInfo::get_sampling_in_t(bin),
                   "test consistency get_sampling_in_t");
#if 0
    check_if_equal(proj_data_info.get_tantheta(bin),
		   proj_data_info.ProjDataInfo::get_tantheta(bin),
		   "test consistency get_tantheta");
#endif
    check_if_equal(
        proj_data_info.get_costheta(bin), proj_data_info.ProjDataInfo::get_costheta(bin), "test consistency get_costheta");

    check_if_equal(proj_data_info.get_costheta(bin),
                   cos(atan(proj_data_info.get_tantheta(bin))),
                   "cross check get_costheta and get_tantheta");
    // set back to usual value
    proj_data_info.set_ring_spacing(old_ring_spacing);
  }

  if (proj_data_info.get_max_ring_difference(0) == proj_data_info.get_min_ring_difference(0)
      && proj_data_info.get_max_ring_difference(1) == proj_data_info.get_min_ring_difference(1)
      && proj_data_info.get_max_ring_difference(2) == proj_data_info.get_min_ring_difference(2))
    {
      // these tests work only without axial compression
      cerr << "\tTest ring pair to segment,ax_pos (span 1)\n";
#ifdef STIR_OPENMP
#  pragma omp parallel for schedule(dynamic)
#endif
      for (int ring1 = 0; ring1 < proj_data_info.get_scanner_ptr()->get_num_rings(); ++ring1)
        for (int ring2 = 0; ring2 < proj_data_info.get_scanner_ptr()->get_num_rings(); ++ring2)
          {
            int segment_num = 0, axial_pos_num = 0;
            check(proj_data_info.get_segment_axial_pos_num_for_ring_pair(segment_num, axial_pos_num, ring1, ring2)
                      == Succeeded::yes,
                  "test if segment,ax_pos_num found for a ring pair");
            check_if_equal(segment_num, ring2 - ring1, "test if segment_num is equal to ring difference\n");
            check_if_equal(axial_pos_num, min(ring2, ring1), "test if segment_num is equal to ring difference\n");

            int check_ring1 = 0, check_ring2 = 0;
            proj_data_info.get_ring_pair_for_segment_axial_pos_num(check_ring1, check_ring2, segment_num, axial_pos_num);
            check_if_equal(ring1, check_ring1, "test ring1 equal after going to segment/ax_pos and returning\n");
            check_if_equal(ring2, check_ring2, "test ring2 equal after going to segment/ax_pos and returning\n");

            const ProjDataInfoCylindrical::RingNumPairs& ring_pairs
                = proj_data_info.get_all_ring_pairs_for_segment_axial_pos_num(segment_num, axial_pos_num);

            check_if_equal(ring_pairs.size(),
                           static_cast<size_t>(1),
                           "test total number of ring-pairs for 1 segment/ax_pos should be 1 for span=1\n");
            check_if_equal(ring1,
                           ring_pairs[0].first,
                           "test ring1 equal after going to segment/ax_pos and returning (version with all ring_pairs)\n");
            check_if_equal(ring2,
                           ring_pairs[0].second,
                           "test ring2 equal after going to segment/ax_pos and returning (version with all ring_pairs)\n");
          }
    }

  cerr << "\tTest ring pair to segment,ax_pos and vice versa (for any axial compression)\n";
  {
#ifdef STIR_OPENMP
#  pragma omp parallel for schedule(dynamic)
#endif
    for (int segment_num = proj_data_info.get_min_segment_num(); segment_num <= proj_data_info.get_max_segment_num();
         ++segment_num)
      for (int axial_pos_num = proj_data_info.get_min_axial_pos_num(segment_num);
           axial_pos_num <= proj_data_info.get_max_axial_pos_num(segment_num);
           ++axial_pos_num)
        {
          const ProjDataInfoCylindrical::RingNumPairs& ring_pairs
              = proj_data_info.get_all_ring_pairs_for_segment_axial_pos_num(segment_num, axial_pos_num);
          for (ProjDataInfoCylindrical::RingNumPairs::const_iterator iter = ring_pairs.begin(); iter != ring_pairs.end(); ++iter)
            {
              int check_segment_num = 0, check_axial_pos_num = 0;
              check(proj_data_info.get_segment_axial_pos_num_for_ring_pair(
                        check_segment_num, check_axial_pos_num, iter->first, iter->second)
                        == Succeeded::yes,
                    "test if segment,ax_pos_num found for a ring pair");
              check_if_equal(check_segment_num, segment_num, "test if segment_num is consistent\n");
              check_if_equal(check_axial_pos_num, axial_pos_num, "test if axial_pos_num is consistent\n");
            }
        }
  }

  test_generic_proj_data_info(proj_data_info);
}

/*!
  The following tests that detection position is affected by the value of DOI
*/
void
ProjDataInfoTests::run_Blocks_DOI_test()
{
  CPUTimer timer;
  auto scannerBlocks_ptr = std::make_shared<Scanner>(Scanner::SAFIRDualRingPrototype);
  scannerBlocks_ptr->set_average_depth_of_interaction(0);
  scannerBlocks_ptr->set_scanner_geometry("BlocksOnCylindrical");
  scannerBlocks_ptr->set_up();

  auto proj_data_info_blocks_doi0_ptr = std::make_shared<ProjDataInfoBlocksOnCylindricalNoArcCorr>();
  proj_data_info_blocks_doi0_ptr = set_blocks_projdata_info<ProjDataInfoBlocksOnCylindricalNoArcCorr>(scannerBlocks_ptr);

  auto scannerBlocksDOI_ptr = std::make_shared<Scanner>(*scannerBlocks_ptr);
  scannerBlocksDOI_ptr->set_average_depth_of_interaction(0.1);
  scannerBlocksDOI_ptr->set_up();

  auto proj_data_info_blocks_doi01_ptr = std::make_shared<ProjDataInfoBlocksOnCylindricalNoArcCorr>();
  proj_data_info_blocks_doi01_ptr = set_blocks_projdata_info<ProjDataInfoBlocksOnCylindricalNoArcCorr>(scannerBlocksDOI_ptr);

  Bin bin;
  LORInAxialAndNoArcCorrSinogramCoordinates<float> lor;

  int Bring1, Bring2, Bdet1, Bdet2, BDring1, BDring2, BDdet1, BDdet2;
  CartesianCoordinate3D<float> b1, b2, bd1, bd2;
  // float doi=scannerBlocksDOI_ptr->get_average_depth_of_interaction();
  timer.reset();
  timer.start();

  for (int seg = proj_data_info_blocks_doi0_ptr->get_min_segment_num();
       seg <= proj_data_info_blocks_doi0_ptr->get_max_segment_num();
       ++seg)
    for (int ax = proj_data_info_blocks_doi0_ptr->get_min_axial_pos_num(seg);
         ax <= proj_data_info_blocks_doi0_ptr->get_max_axial_pos_num(seg);
         ++ax)
      for (int view = 0; view <= proj_data_info_blocks_doi0_ptr->get_max_view_num(); view++)
        for (int tang = proj_data_info_blocks_doi0_ptr->get_min_tangential_pos_num();
             tang <= proj_data_info_blocks_doi0_ptr->get_max_tangential_pos_num();
             ++tang)
          {
            bin.segment_num() = seg;
            bin.axial_pos_num() = ax;
            bin.view_num() = view;
            bin.tangential_pos_num() = tang;

            //                check det_pos instead
            proj_data_info_blocks_doi0_ptr->get_det_pair_for_bin(Bdet1, Bring1, Bdet2, Bring2, bin);
            proj_data_info_blocks_doi01_ptr->get_det_pair_for_bin(BDdet1, BDring1, BDdet2, BDring2, bin);

            proj_data_info_blocks_doi0_ptr->get_LOR(lor, bin);
            set_tolerance(10E-4);

            check_if_equal(Bdet1, BDdet1, "");
            check_if_equal(Bdet2, BDdet2, "");
            check_if_equal(Bring1, BDring1, "");
            check_if_equal(Bring2, BDring2, "");

            //                checkcartesian coordinates of detectors
            proj_data_info_blocks_doi0_ptr->find_cartesian_coordinates_of_detection(b1, b2, bin);
            proj_data_info_blocks_doi01_ptr->find_cartesian_coordinates_of_detection(bd1, bd2, bin);

            //               set_tolerance(10E-2);
            check(b1 != bd1, "detector position should be different with different DOIs");
            check(b2 != bd2, "detector position should be different with different DOIs");
            // TODO improve check here
          }
  timer.stop();
  std::cerr << "-- CPU Time " << timer.value() << '\n';
}

/*!
  The following tests the consistency of coordinates obtained with a cilindrical scanner
  and those of a blocks on cylindrical scanner. For this test, a scanner with 4 rings, 2
  detector per block and 2 blcs per bucket is used axially. However, transaxially we have
  1 crystal per block a 1 block per bucket

  In this function, an extra test is performed to check that a roundtrip
  transformation: detector_ID->cartesian_coord_detection_pos->detector_ID
  provides the same as the starting point
*/
void
ProjDataInfoTests::run_coordinate_test()
{
  CPUTimer timer;
  auto scannerBlocks_ptr = std::make_shared<Scanner>(Scanner::SAFIRDualRingPrototype);
  scannerBlocks_ptr->set_num_axial_crystals_per_block(2);
  scannerBlocks_ptr->set_axial_block_spacing(scannerBlocks_ptr->get_axial_crystal_spacing()
                                             * scannerBlocks_ptr->get_num_axial_crystals_per_block());
  scannerBlocks_ptr->set_num_transaxial_crystals_per_block(1);
  scannerBlocks_ptr->set_num_axial_blocks_per_bucket(2);
  scannerBlocks_ptr->set_num_transaxial_blocks_per_bucket(1);
  scannerBlocks_ptr->set_transaxial_block_spacing(scannerBlocks_ptr->get_transaxial_crystal_spacing()
                                                  * scannerBlocks_ptr->get_num_transaxial_crystals_per_block());
  scannerBlocks_ptr->set_num_rings(4);

  scannerBlocks_ptr->set_scanner_geometry("BlocksOnCylindrical");
  scannerBlocks_ptr->set_up();

  auto scannerCyl_ptr = std::make_shared<Scanner>(Scanner::SAFIRDualRingPrototype);
  scannerCyl_ptr->set_num_axial_crystals_per_block(2);
  scannerCyl_ptr->set_axial_block_spacing(scannerCyl_ptr->get_axial_crystal_spacing()
                                          * scannerCyl_ptr->get_num_axial_crystals_per_block());
  scannerCyl_ptr->set_transaxial_block_spacing(scannerCyl_ptr->get_transaxial_crystal_spacing()
                                               * scannerCyl_ptr->get_num_transaxial_crystals_per_block());
  scannerCyl_ptr->set_num_transaxial_crystals_per_block(1);
  scannerCyl_ptr->set_num_axial_blocks_per_bucket(2);
  scannerCyl_ptr->set_num_transaxial_blocks_per_bucket(1);

  scannerCyl_ptr->set_num_rings(4);
  scannerCyl_ptr->set_scanner_geometry("Cylindrical");
  scannerCyl_ptr->set_up();

  auto proj_data_info_blocks_ptr = std::make_shared<ProjDataInfoBlocksOnCylindricalNoArcCorr>();
  proj_data_info_blocks_ptr = set_blocks_projdata_info<ProjDataInfoBlocksOnCylindricalNoArcCorr>(scannerBlocks_ptr);

  auto proj_data_info_cyl_ptr = std::make_shared<ProjDataInfoCylindricalNoArcCorr>();
  proj_data_info_cyl_ptr = set_blocks_projdata_info<ProjDataInfoCylindricalNoArcCorr>(scannerCyl_ptr);
  Bin bin, binRT;

  int Bring1, Bring2, Bdet1, Bdet2, Cring1, Cring2, Cdet1, Cdet2;
  int RTring1, RTring2, RTdet1, RTdet2;
  CartesianCoordinate3D<float> b1, b2, c1, c2, roundt1, roundt2;
  timer.reset();
  timer.start();
  LORInAxialAndNoArcCorrSinogramCoordinates<float> lorB;
  LORInAxialAndNoArcCorrSinogramCoordinates<float> lorC;

  LORInAxialAndNoArcCorrSinogramCoordinates<float> lorC1, lorCn;

  for (int seg = proj_data_info_blocks_ptr->get_min_segment_num(); seg <= proj_data_info_blocks_ptr->get_max_segment_num(); ++seg)
    for (int ax = proj_data_info_blocks_ptr->get_min_axial_pos_num(seg);
         ax <= proj_data_info_blocks_ptr->get_max_axial_pos_num(seg);
         ++ax)
      for (int view = 0; view <= proj_data_info_blocks_ptr->get_max_view_num(); view++)
        for (int tang = proj_data_info_blocks_ptr->get_min_tangential_pos_num();
             tang <= proj_data_info_blocks_ptr->get_max_tangential_pos_num();
             ++tang)
          {
            bin.segment_num() = seg;
            bin.axial_pos_num() = ax;
            bin.view_num() = view;
            bin.tangential_pos_num() = tang;

            proj_data_info_cyl_ptr->get_LOR(lorC, bin);
            proj_data_info_blocks_ptr->get_LOR(lorB, bin);

            const int num_detectors = proj_data_info_cyl_ptr->get_scanner_ptr()->get_num_detectors_per_ring();

            int det_num1 = 0, det_num2 = 0;
            proj_data_info_cyl_ptr->get_det_num_pair_for_view_tangential_pos_num(
                det_num1, det_num2, bin.view_num(), bin.tangential_pos_num());

            float phi;
            phi = static_cast<float>((det_num1 + det_num2) * _PI / num_detectors - _PI / 2
                                     + proj_data_info_cyl_ptr->get_azimuthal_angle_offset());

            lorC1 = LORInAxialAndNoArcCorrSinogramCoordinates<float>(lorC.z1(),
                                                                     lorC.z2(),
                                                                     phi, // lorC.phi(),
                                                                     lorC.beta(),
                                                                     proj_data_info_cyl_ptr->get_ring_radius());

            const float old_phi = proj_data_info_cyl_ptr->get_phi(bin);
            if (fabs(phi - old_phi) >= 2 * _PI / num_detectors)
              {
                // float ang=2*_PI/num_detectors/2;
                //                  warning("view %d old_phi %g new_phi %g\n",bin.view_num(), old_phi, phi);

                lorC1 = LORInAxialAndNoArcCorrSinogramCoordinates<float>(lorC.z2(),
                                                                         lorC.z1(),
                                                                         phi, // lorC.phi(),
                                                                         -lorC.beta(),
                                                                         proj_data_info_cyl_ptr->get_ring_radius());
              }
            //                check det_pos instead
            proj_data_info_blocks_ptr->get_det_pair_for_bin(Bdet1, Bring1, Bdet2, Bring2, bin);
            proj_data_info_cyl_ptr->get_det_pair_for_bin(Cdet1, Cring1, Cdet2, Cring2, bin);

            set_tolerance(10E-4);

            check_if_equal(Bdet1, Cdet1, "");
            check_if_equal(Bdet2, Cdet2, "");
            check_if_equal(Bring1, Cring1, "");
            check_if_equal(Bring2, Cring2, "");

            //                test round trip from detector ID to coordinates and from cordinates to detecto IDs
            proj_data_info_blocks_ptr->find_cartesian_coordinates_given_scanner_coordinates(
                roundt1, roundt2, Bring1, Bring2, Bdet1, Bdet2);
            proj_data_info_blocks_ptr->find_bin_given_cartesian_coordinates_of_detection(binRT, roundt1, roundt2);
            proj_data_info_blocks_ptr->get_det_pair_for_bin(RTdet1, RTring1, RTdet2, RTring2, bin);

            check_if_equal(Bdet1, RTdet1, "Roundtrip from detector A ID to coordinates and from cordinates to detector A ID");
            check_if_equal(Bdet2, RTdet2, "Roundtrip from detector B ID to coordinates and from cordinates to detector B ID");
            check_if_equal(Bring1, RTring1, "Roundtrip from ring A ID to coordinates and from cordinates to ring A ID");
            check_if_equal(Bring2, RTring2, "Roundtrip from ring B ID to coordinates and from cordinates to ring B ID");

            //                checkcartesian coordinates of detectors
            proj_data_info_cyl_ptr->find_cartesian_coordinates_of_detection(c1, c2, bin);
            proj_data_info_blocks_ptr->find_cartesian_coordinates_of_detection(b1, b2, bin);

            check_if_equal(b1, c1, "");
            check_if_equal(b2, c2, "");

            set_tolerance(10E-3);

            if (abs(lorB.phi() - lorC1.phi()) < tolerance)
              {

                check_if_equal(proj_data_info_blocks_ptr->get_s(bin),
                               lorB.s(),
                               "A get_s() from projdata is different from Block on Cylindrical LOR.s()");

                check_if_equal(lorB.s(),
                               lorC1.s(),
                               "tang_pos=" + std::to_string(tang) + " PHI-C=" + std::to_string(lorC1.phi())
                                   + " PHI-B=" + std::to_string(lorB.phi()) + " view=" + std::to_string(view)
                                   + " Atest if BlocksOnCylindrical LOR.s is the same as the LOR produced by Cylindrical"); //)

                check_if_equal(lorB.beta(),
                               lorC1.beta(),
                               "tang_pos=" + std::to_string(tang) + " ax_pos=" + std::to_string(ax)
                                   + " segment=" + std::to_string(seg) + " view=" + std::to_string(view)
                                   + " test if BlocksOnCylindrical LOR.beta is the same as the LOR produced by Cylindrical");
                check_if_equal(lorB.z1(),
                               lorC1.z1(),
                               "tang_pos=" + std::to_string(tang) + " ax_pos=" + std::to_string(ax)
                                   + " segment=" + std::to_string(seg) + " view=" + std::to_string(view)
                                   + " test if BlocksOnCylindrical LOR.z1 is the same as the LOR produced by Cylindrical");
                check_if_equal(lorB.z2(),
                               lorC1.z2(),
                               "tang_pos=" + std::to_string(tang) + " ax_pos=" + std::to_string(ax)
                                   + " segment=" + std::to_string(seg) + " view=" + std::to_string(view)
                                   + " test if BlocksOnCylindrical LOR.z2 is the same as the LOR produced by Cylindrical");

                //                TODO: fix problem with interleaving when calculating Phi
              }
            else if (abs(lorB.phi() - lorC1.phi()) + _PI < tolerance || abs(lorB.phi() - lorC1.phi()) - _PI < tolerance)
              {

                check_if_equal(proj_data_info_blocks_ptr->get_s(bin),
                               lorB.s(),
                               "B get_s() from projdata is different from Block on Cylindrical LOR.s()");
                check_if_equal(proj_data_info_blocks_ptr->get_phi(bin),
                               phi,
                               "B get_phi() from projdata Cylinder is different from Block on Cylindrical");

                check_if_equal(lorB.s(),
                               -lorC1.s(),
                               "tang_pos=" + std::to_string(tang) + " PHYC=" + std::to_string(lorC1.phi())
                                   + " PHIB=" + std::to_string(lorB.phi()) + " view=" + std::to_string(view)
                                   + " Btest if BlocksOnCylindrical LOR.s is the same as the LOR produced by Cylindrical"); //)
                check_if_equal(lorB.beta(),
                               -lorC1.beta(),
                               " Btest if BlocksOnCylindrical LOR.beta is the same as the LOR produced by Cylindrical");
                check_if_equal(lorB.z1(),
                               lorC1.z2(),
                               "tang_pos=" + std::to_string(tang) + " ax_pos=" + std::to_string(ax)
                                   + " segment=" + std::to_string(seg) + " view=" + std::to_string(view)
                                   + " Btest if BlocksOnCylindrical LOR.z1 is the same as the LOR produced by Cylindrical");
                check_if_equal(
                    lorB.z2(), lorC1.z1(), " Btest if BlocksOnCylindrical LOR.z2 is the same as the LOR produced by Cylindrical");
              }
            else
              {
                check(false, "phi is different");
              }
            check_if_equal(proj_data_info_blocks_ptr->get_m(bin), proj_data_info_cyl_ptr->get_m(bin), " test get_m Cylindrical");
          }
  timer.stop();
  std::cerr << "-- CPU Time " << timer.value() << '\n';
}

/*!
  The following test is similar to the above but for a scanner that has multiple rings and
  multiple detectors per block.
*/
void
ProjDataInfoTests::run_coordinate_test_for_realistic_scanner()
{
  CPUTimer timer;
  auto scannerBlocks_ptr = std::make_shared<Scanner>(Scanner::SAFIRDualRingPrototype);
  scannerBlocks_ptr->set_axial_block_spacing(scannerBlocks_ptr->get_axial_crystal_spacing()
                                             * scannerBlocks_ptr->get_num_axial_crystals_per_block());
  scannerBlocks_ptr->set_transaxial_block_spacing(scannerBlocks_ptr->get_transaxial_crystal_spacing()
                                                  * scannerBlocks_ptr->get_num_transaxial_crystals_per_block());

  scannerBlocks_ptr->set_scanner_geometry("BlocksOnCylindrical");
  scannerBlocks_ptr->set_up();

  auto scannerCyl_ptr = std::make_shared<Scanner>(Scanner::SAFIRDualRingPrototype);
  scannerCyl_ptr->set_axial_block_spacing(scannerCyl_ptr->get_axial_crystal_spacing()
                                          * scannerCyl_ptr->get_num_axial_crystals_per_block());
  scannerCyl_ptr->set_transaxial_block_spacing(scannerCyl_ptr->get_transaxial_crystal_spacing()
                                               * scannerCyl_ptr->get_num_transaxial_crystals_per_block());

  scannerCyl_ptr->set_scanner_geometry("Cylindrical");
  scannerCyl_ptr->set_up();

  auto proj_data_info_blocks_ptr = std::make_shared<ProjDataInfoBlocksOnCylindricalNoArcCorr>();
  proj_data_info_blocks_ptr = set_blocks_projdata_info<ProjDataInfoBlocksOnCylindricalNoArcCorr>(scannerBlocks_ptr);

  auto proj_data_info_cyl_ptr = std::make_shared<ProjDataInfoCylindricalNoArcCorr>();
  proj_data_info_cyl_ptr = set_blocks_projdata_info<ProjDataInfoCylindricalNoArcCorr>(scannerCyl_ptr);

  Bin bin;

  int Bring1, Bring2, Bdet1, Bdet2, Cring1, Cring2, Cdet1, Cdet2;
  CartesianCoordinate3D<float> b1, b2, c1, c2;

  //    estimate the angle covered by half bucket, csi
  float csi = _PI / scannerBlocks_ptr->get_num_transaxial_buckets();
  //    distance between the center of the scannner and the first crystal in the bucket, r=Reffective/cos(csi)
  float r = scannerBlocks_ptr->get_effective_ring_radius() / cos(csi);
  float max_tolerance = abs(scannerBlocks_ptr->get_effective_ring_radius() - r);

  timer.reset();
  timer.start();
  LORInAxialAndNoArcCorrSinogramCoordinates<float> lorB;
  LORInAxialAndNoArcCorrSinogramCoordinates<float> lorC;

  LORInAxialAndNoArcCorrSinogramCoordinates<float> lorC1, lorCn;

  for (int seg = proj_data_info_blocks_ptr->get_min_segment_num(); seg <= proj_data_info_blocks_ptr->get_max_segment_num(); ++seg)
    for (int ax = proj_data_info_blocks_ptr->get_min_axial_pos_num(seg);
         ax <= proj_data_info_blocks_ptr->get_max_axial_pos_num(seg);
         ++ax)
      for (int view = 0; view <= proj_data_info_blocks_ptr->get_max_view_num(); view++)
        for (int tang = proj_data_info_blocks_ptr->get_min_tangential_pos_num();
             tang <= proj_data_info_blocks_ptr->get_max_tangential_pos_num();
             ++tang)
          {
            bin.segment_num() = seg;
            bin.axial_pos_num() = ax;
            bin.view_num() = view;
            bin.tangential_pos_num() = tang;

            proj_data_info_cyl_ptr->get_LOR(lorC, bin);
            proj_data_info_blocks_ptr->get_LOR(lorB, bin);

            int det_num1 = 0, det_num2 = 0;
            proj_data_info_cyl_ptr->get_det_num_pair_for_view_tangential_pos_num(
                det_num1, det_num2, bin.view_num(), bin.tangential_pos_num());

            //                check det_pos instead
            proj_data_info_blocks_ptr->get_det_pair_for_bin(Bdet1, Bring1, Bdet2, Bring2, bin);
            proj_data_info_cyl_ptr->get_det_pair_for_bin(Cdet1, Cring1, Cdet2, Cring2, bin);

            set_tolerance(10E-4);

            check_if_equal(Bdet1, Cdet1, "");
            check_if_equal(Bdet2, Cdet2, "");
            check_if_equal(Bring1, Cring1, "");
            check_if_equal(Bring2, Cring2, "");

            //                check cartesian coordinates of detectors
            proj_data_info_cyl_ptr->find_cartesian_coordinates_of_detection(c1, c2, bin);
            proj_data_info_blocks_ptr->find_cartesian_coordinates_of_detection(b1, b2, bin);

            // we expect to be differences of the order of the mm in x and y due to the difference in geometry

            set_tolerance(max_tolerance);

            check_if_equal(b1.y(), c1.y(), " checking cartesian coordinate y1");
            check_if_equal(b2.y(), c2.y(), " checking cartesian coordinate y2");
            check_if_equal(b1.x(), c1.x(), " checking cartesian coordinate x1");
            check_if_equal(b2.x(), c2.x(), " checking cartesian coordinate x2");

            /*!calculate the max axial tolerance, the difference between m (blocks vs cylindrical) happens when the point A is at
             * the beginning of a transaxial bucket and the point B is in the middle of the bucket (which is the point where the
             * radius of circle and the radius of the poligon have the biggest difference
             * !*/
            float psi = atan(abs(lorB.z2() - lorB.z1()) / r);
            float max_ax_tolerance = abs(scannerBlocks_ptr->get_effective_ring_radius() - r) / cos(psi);
            set_tolerance(max_ax_tolerance);
            check_if_equal(b1.z(), c1.z(), " checking cartesian coordinate z1");
            check_if_equal(b2.z(), c2.z(), " checking cartesian coordinate z2");
            check_if_equal(proj_data_info_blocks_ptr->get_m(bin), proj_data_info_cyl_ptr->get_m(bin), " test get_m Cylindrical");
          }
  timer.stop();
  std::cerr << "-- CPU Time " << timer.value() << '\n';
}

/*!
  The following tests the function get_s() for the BlockOnCylindrical case. the first test
  checks that all lines passing for the center provide s=0. The second test checks that
  parallel lines are always at the same angle phi, and that the step between consecutive
  lines is the same and equal to the one calculated geometrically.
*/
void
ProjDataInfoTests::run_lor_get_s_test()
{
  CPUTimer timer;
  auto scannerBlocks_ptr = std::make_shared<Scanner>(Scanner::SAFIRDualRingPrototype);
  scannerBlocks_ptr->set_axial_block_spacing(scannerBlocks_ptr->get_axial_crystal_spacing()
                                             * scannerBlocks_ptr->get_num_axial_crystals_per_block());
  scannerBlocks_ptr->set_transaxial_block_spacing(scannerBlocks_ptr->get_transaxial_crystal_spacing()
                                                  * scannerBlocks_ptr->get_num_transaxial_crystals_per_block());

  scannerBlocks_ptr->set_scanner_geometry("BlocksOnCylindrical");
  scannerBlocks_ptr->set_up();

  auto scannerCyl_ptr = std::make_shared<Scanner>(Scanner::SAFIRDualRingPrototype);
  scannerCyl_ptr->set_axial_block_spacing(scannerCyl_ptr->get_axial_crystal_spacing()
                                          * scannerCyl_ptr->get_num_axial_crystals_per_block());
  scannerCyl_ptr->set_transaxial_block_spacing(scannerCyl_ptr->get_transaxial_crystal_spacing()
                                               * scannerCyl_ptr->get_num_transaxial_crystals_per_block());

  scannerCyl_ptr->set_scanner_geometry("Cylindrical");
  scannerCyl_ptr->set_up();

  auto proj_data_info_blocks_ptr = std::make_shared<ProjDataInfoBlocksOnCylindricalNoArcCorr>();
  proj_data_info_blocks_ptr = set_blocks_projdata_info<ProjDataInfoBlocksOnCylindricalNoArcCorr>(scannerBlocks_ptr);

  auto proj_data_info_cyl_ptr = std::make_shared<ProjDataInfoCylindricalNoArcCorr>();
  proj_data_info_cyl_ptr = set_blocks_projdata_info<ProjDataInfoCylindricalNoArcCorr>(scannerCyl_ptr);
  // select detection position 1

  LORInAxialAndNoArcCorrSinogramCoordinates<float> lorB;
  LORInAxialAndNoArcCorrSinogramCoordinates<float> lorC;
  int Cring1, Cring2, Cdet1, Cdet2;
  Bin bin;
  //    Det<> pos1(0,0,0);
  set_tolerance(10E-4);
  for (int i = 0; i < scannerCyl_ptr->get_num_detectors_per_ring(); i++)
    {

      Cring1 = 0;
      Cdet1 = i;
      Cring2 = 0;
      Cdet2 = scannerCyl_ptr->get_num_detectors_per_ring() / 2 + Cdet1;
      if (Cdet2 >= scannerCyl_ptr->get_num_detectors_per_ring())
        Cdet2 = Cdet1 - scannerCyl_ptr->get_num_detectors_per_ring() / 2;
      const DetectionPositionPair<> det_pos_pair(DetectionPosition<>(Cdet1, Cring1), DetectionPosition<>(Cdet2, Cring2));
      proj_data_info_cyl_ptr->get_bin_for_det_pos_pair(bin, det_pos_pair);
      proj_data_info_cyl_ptr->get_LOR(lorC, bin);

      proj_data_info_blocks_ptr->get_bin_for_det_pos_pair(bin, det_pos_pair);
      proj_data_info_blocks_ptr->get_LOR(lorB, bin);

      check_if_equal(
          0., lorC.s(), std::to_string(i) + " Cylinder get_s() should be zero when the LOR passes at the center of the scanner");
      check_if_equal(
          0., lorB.s(), std::to_string(i) + " Blocks get_s() should be zero when the LOR passes at the center of the scanner");
    }

  //    Check get_s() (for BlocksOnCylindrical) when the line is at a given angle. We consider two blocks at a relative angle of
  //    90 degrees the angle covered by the detectors is 120 (each block is 30 degrees and the detector are 4 blocks apart). The
  //    following LOR will be obtained by increasing detID1 and decreasing detID2 so that they are always parallel.
  //

  //    Let's calculate the relative ID difference between block1 and block2 in coincidence
  //    num_det_per_ring:2PI=det_id_diff:PI/2
  int det_id_diff = scannerBlocks_ptr->get_num_detectors_per_ring() / 4;
  int Ctb = scannerCyl_ptr->get_num_transaxial_crystals_per_block();
  float transaxial_crystal_spacing = scannerBlocks_ptr->get_transaxial_crystal_spacing();
  float prev_s = 0;
  float prev_phi = 0;
  for (int i = 0; i < scannerCyl_ptr->get_num_transaxial_crystals_per_block(); i++)
    {

      Cring1 = 0;
      Cdet1 = i + 2 * Ctb;
      Cring2 = 0;
      Cdet2 = 2 * Ctb + det_id_diff + Ctb - 1 - i;

      const DetectionPositionPair<> det_pos_pair(DetectionPosition<>(Cdet1, Cring1), DetectionPosition<>(Cdet2, Cring2));

      proj_data_info_blocks_ptr->get_bin_for_det_pos_pair(bin, det_pos_pair);
      proj_data_info_blocks_ptr->get_LOR(lorB, bin);
      /*float R=block_trans_spacing*(sin(_PI*5/12)+sin(_PI/4)+sin(_PI/12));
        float s=R*cos(_PI/3)+
              transaxial_crystal_spacing/2*sin(_PI/4)+
              (i)*transaxial_crystal_spacing*sin(_PI/4);*/

      float s_step = transaxial_crystal_spacing * sin(_PI / 4);

      //        the following fails at the moment
      //        check_if_equal(s, lorB.s(),std::to_string(i)+ " Blocks get_s()  is different");
      //        the first value we expect to be different
      set_tolerance(10E-3);
      if (i > 0)
        {
          check_if_equal(s_step, lorB.s() - prev_s, std::to_string(i) + " Blocks get_s() the step is different");
          check_if_equal(
              0.F, lorB.phi() - prev_phi, " Blocks get_phi() should be always the same as we are considering parallel LORs");
        }
      prev_s = lorB.s();
      prev_phi = lorB.phi();
    }
}

/*!
  \ingroup test
  \brief Test class for ProjDataInfoCylindricalArcCorr
*/

class ProjDataInfoCylindricalArcCorrTests : public ProjDataInfoCylindricalTests
{
public:
  void run_tests() override;
};

void
ProjDataInfoCylindricalArcCorrTests::run_tests()

{

  std::cerr << "-------- Testing ProjData Geometry --------\n";

  std::cerr << "-------- Testing DOI for blocks --------\n";
  run_Blocks_DOI_test();
  std::cerr << "-------- Testing coordinates --------\n";
  run_lor_get_s_test();
  run_coordinate_test();
  run_coordinate_test_for_realistic_scanner();

  cerr << "-------- Testing ProjDataInfoCylindricalArcCorr --------\n";
  {
    // Test on the empty constructor

    ProjDataInfoCylindricalArcCorr ob1;

    // Test on set.* & get.* + constructor
    const float test_tangential_sampling = 1.5;
    // const float test_azimuthal_angle_sampling = 10.1;

    ob1.set_tangential_sampling(test_tangential_sampling);
    // Set_azimuthal_angle_sampling
    // ob1.set_azimuthal_angle_sampling(test_azimuthal_angle_sampling);

    check_if_equal(ob1.get_tangential_sampling(), test_tangential_sampling, "test on tangential_sampling");
    // check_if_zero( ob1.get_azimuthal_angle_sampling() - test_azimuthal_angle_sampling, " test on azimuthal_angle_sampling");
  }
  {
    shared_ptr<Scanner> scanner_ptr(new Scanner(Scanner::E953));

    VectorWithOffset<int> num_axial_pos_per_segment(-1, 1);
    VectorWithOffset<int> min_ring_diff(-1, 1);
    VectorWithOffset<int> max_ring_diff(-1, 1);
    // simulate span=3 for segment 0, span=1 for segment 2
    num_axial_pos_per_segment[-1] = 14;
    num_axial_pos_per_segment[0] = 31;
    num_axial_pos_per_segment[1] = 14;
    // KT 28/11/2001 corrected typo (bug): min_ring_diff[-1] was initialised twice, and max_ring_diff[-1] wasn't
    min_ring_diff[-1] = max_ring_diff[-1] = -2;
    min_ring_diff[0] = -1;
    max_ring_diff[0] = 1;
    min_ring_diff[+1] = max_ring_diff[+1] = +2;
    const int num_views = 96;
    const int num_tangential_poss = 128;

    const float bin_size = 1.2F;

    // Test on the constructor
    ProjDataInfoCylindricalArcCorr ob2(
        scanner_ptr, bin_size, num_axial_pos_per_segment, min_ring_diff, max_ring_diff, num_views, num_tangential_poss);

    check_if_equal(ob2.get_tangential_sampling(), bin_size, "test on tangential_sampling");
    check_if_equal(ob2.get_azimuthal_angle_sampling(), _PI / num_views, " test on azimuthal_angle_sampling");
    check_if_equal(ob2.get_axial_sampling(1), scanner_ptr->get_ring_spacing(), "test on axial_sampling");
    check_if_equal(ob2.get_axial_sampling(0), scanner_ptr->get_ring_spacing() / 2, "test on axial_sampling for segment0");

    {
      // segment 0
      Bin bin(0, 10, 10, 20);
      float theta = ob2.get_tantheta(bin);
      float phi = ob2.get_phi(bin);
      // Get t
      float t = ob2.get_t(bin);
      //! Get s
      float s = ob2.get_s(bin);

      check_if_equal(theta, 0.F, "test on get_tantheta, seg 0");
      check_if_equal(phi, 10 * ob2.get_azimuthal_angle_sampling() + ob2.get_azimuthal_angle_offset(), " get_phi , seg 0");
      // KT 25/10/2000 adjust to new convention
      const float ax_pos_origin = (ob2.get_min_axial_pos_num(0) + ob2.get_max_axial_pos_num(0)) / 2.F;
      check_if_equal(t, (10 - ax_pos_origin) * ob2.get_axial_sampling(0), "get_t, seg 0");
      check_if_equal(s, 20 * ob2.get_tangential_sampling(), "get_s, seg 0");
    }
    {
      // Segment 1
      Bin bin(1, 10, 10, 20);
      float theta = ob2.get_tantheta(bin);
      float phi = ob2.get_phi(bin);
      // Get t
      float t = ob2.get_t(bin);
      // Get s
      float s = ob2.get_s(bin);

      float thetatest = 2 * ob2.get_axial_sampling(1) / (2 * sqrt(square(scanner_ptr->get_effective_ring_radius()) - square(s)));

      check_if_equal(theta, thetatest, "test on get_tantheta, seg 1");
      check_if_equal(phi, 10 * ob2.get_azimuthal_angle_sampling() + ob2.get_azimuthal_angle_offset(), " get_phi , seg 1");
      // KT 25/10/2000 adjust to new convention
      const float ax_pos_origin = (ob2.get_min_axial_pos_num(1) + ob2.get_max_axial_pos_num(1)) / 2.F;
      check_if_equal(t, (10 - ax_pos_origin) / sqrt(1 + square(thetatest)) * ob2.get_axial_sampling(1), "get_t, seg 1");
      check_if_equal(s, 20 * ob2.get_tangential_sampling(), "get_s, seg 1");
    }

#if 0
  // disabled to get noninteractive test
  michelogram(ob2);
  cerr << endl;
#endif
  }

#if 0    

  // disabled to get noninteractive test
  {
    shared_ptr<Scanner> scanner_ptr = new Scanner(Scanner::E953);
    
    shared_ptr<ProjDataInfo> proj_data_info_ptr =
      ProjDataInfo::construct_proj_data_info(scanner_ptr,
		                    /*span*/1, 10,/*views*/ 96, /*tang_pos*/128, /*arc_corrected*/ true);
    michelogram(dynamic_cast<const ProjDataInfoCylindrical&>(*proj_data_info_ptr));
    cerr << endl;
  }
 {
    shared_ptr<Scanner> scanner_ptr = new Scanner(Scanner::E953);
    
    shared_ptr<ProjDataInfo> proj_data_info_ptr =
      ProjDataInfo::construct_proj_data_info(scanner_ptr,
		                    /*span*/7, 10,/*views*/ 96, /*tang_pos*/128, /*arc_corrected*/ true);
    michelogram(dynamic_cast<const ProjDataInfoCylindrical&>(*proj_data_info_ptr));
    cerr << endl;
  }
#endif

  shared_ptr<Scanner> scanner_ptr(new Scanner(Scanner::E953));
  cerr << "Tests with proj_data_info without mashing and axial compression\n\n";
  // Note: test without axial compression requires that all ring differences
  // are in some segment, so use maximum ring difference
  shared_ptr<ProjDataInfo> proj_data_info_ptr(
      ProjDataInfo::construct_proj_data_info(scanner_ptr,
                                             /*span*/ 1,
                                             scanner_ptr->get_num_rings() - 1,
                                             /*views*/ scanner_ptr->get_num_detectors_per_ring() / 2,
                                             /*tang_pos*/ 64,
                                             /*arc_corrected*/ true));
  test_cylindrical_proj_data_info(dynamic_cast<ProjDataInfoCylindricalArcCorr&>(*proj_data_info_ptr));

  cerr << "\nTests with proj_data_info with mashing and axial compression (span 5)\n\n";
  proj_data_info_ptr = ProjDataInfo::construct_proj_data_info(scanner_ptr,
                                                              /*span*/ 5,
                                                              scanner_ptr->get_num_rings() - 1,
                                                              /*views*/ scanner_ptr->get_num_detectors_per_ring() / 2 / 8,
                                                              /*tang_pos*/ 64,
                                                              /*arc_corrected*/ true);
  test_cylindrical_proj_data_info(dynamic_cast<ProjDataInfoCylindricalArcCorr&>(*proj_data_info_ptr));

  cerr << "\nTests with proj_data_info with mashing and axial compression (span 4)\n\n";
  proj_data_info_ptr = ProjDataInfo::construct_proj_data_info(scanner_ptr,
                                                              /*span*/ 4,
                                                              scanner_ptr->get_num_rings() - 1,
                                                              /*views*/ scanner_ptr->get_num_detectors_per_ring() / 2 / 8,
                                                              /*tang_pos*/ 64,
                                                              /*arc_corrected*/ true);
#if 0
	// disabled to get noninteractive test
	michelogram(dynamic_cast<ProjDataInfoCylindrical&>(*proj_data_info_ptr));
	cerr << endl;
#endif
  test_cylindrical_proj_data_info(dynamic_cast<ProjDataInfoCylindricalArcCorr&>(*proj_data_info_ptr));
}

/*!
  \ingroup test
  \brief Test class for ProjDataInfoCylindricalNoArcCorr
*/

class ProjDataInfoCylindricalNoArcCorrTests : public ProjDataInfoCylindricalTests
{
public:
  void run_tests() override;

private:
  void test_proj_data_info(ProjDataInfoCylindricalNoArcCorr& proj_data_info);
};

void
ProjDataInfoCylindricalNoArcCorrTests::run_tests()
{
  cerr << "\n-------- Testing ProjDataInfoCylindricalNoArcCorr --------\n";
  shared_ptr<Scanner> scanner_ptr(new Scanner(Scanner::E953));
  cerr << "Tests with proj_data_info without mashing and axial compression\n\n";
  // Note: test without axial compression requires that all ring differences
  // are in some segment, so use maximum ring difference
  shared_ptr<ProjDataInfo> proj_data_info_ptr(
      ProjDataInfo::construct_proj_data_info(scanner_ptr,
                                             /*span*/ 1,
                                             scanner_ptr->get_num_rings() - 1,
                                             /*views*/ scanner_ptr->get_num_detectors_per_ring() / 2,
                                             /*tang_pos*/ 64,
                                             /*arc_corrected*/ false));
#ifndef STIR_TOF_DEBUG // disable these for speed of testing
  test_proj_data_info(dynamic_cast<ProjDataInfoCylindricalNoArcCorr&>(*proj_data_info_ptr));

  cerr << "\nTests with proj_data_info with mashing and axial compression (span 5)\n\n";
  proj_data_info_ptr = ProjDataInfo::construct_proj_data_info(scanner_ptr,
                                                              /*span*/ 5,
                                                              scanner_ptr->get_num_rings() - 1,
                                                              /*views*/ scanner_ptr->get_num_detectors_per_ring() / 2 / 8,
                                                              /*tang_pos*/ 64,
                                                              /*arc_corrected*/ false);
  test_proj_data_info(dynamic_cast<ProjDataInfoCylindricalNoArcCorr&>(*proj_data_info_ptr));

  cerr << "\nTests with proj_data_info with mashing and axial compression (span 2)\n\n";
  proj_data_info_ptr = ProjDataInfo::construct_proj_data_info(scanner_ptr,
                                                              /*span*/ 2,
                                                              scanner_ptr->get_num_rings() - 7,
                                                              /*views*/ scanner_ptr->get_num_detectors_per_ring() / 2 / 8,
                                                              /*tang_pos*/ 64,
                                                              /*arc_corrected*/ false);
  test_proj_data_info(dynamic_cast<ProjDataInfoCylindricalNoArcCorr&>(*proj_data_info_ptr));
#endif // STIR_TOF_DEBUG
  cerr << "\nTests with proj_data_info with time-of-flight\n\n";
  shared_ptr<Scanner> scanner_tof_ptr(new Scanner(Scanner::Discovery690));
  proj_data_info_ptr = ProjDataInfo::construct_proj_data_info(scanner_tof_ptr,
                                                              /*span*/ 11,
                                                              scanner_tof_ptr->get_num_rings() - 1,
                                                              /*views*/ scanner_tof_ptr->get_num_detectors_per_ring() / 2,
                                                              /*tang_pos*/ 64,
                                                              /*arc_corrected*/ false,
                                                              /*tof_mashing*/ 5);
  test_proj_data_info(dynamic_cast<ProjDataInfoCylindricalNoArcCorr&>(*proj_data_info_ptr));
}

void
ProjDataInfoCylindricalNoArcCorrTests::test_proj_data_info(ProjDataInfoCylindricalNoArcCorr& proj_data_info)
{
  test_cylindrical_proj_data_info(proj_data_info);

  const int num_detectors = proj_data_info.get_scanner_ptr()->get_num_detectors_per_ring();

#ifndef TEST_ONLY_GET_BIN
  if (proj_data_info.get_view_mashing_factor() == 1)
    {
      // these tests work only without mashing

      cerr << "\n\tTest code for sinogram <-> detector conversions.";

#  ifdef STIR_OPENMP
#    pragma omp parallel for schedule(dynamic)
#  endif
      for (int det_num_a = 0; det_num_a < num_detectors; det_num_a++)
        for (int det_num_b = 0; det_num_b < num_detectors; det_num_b++)
          {
            int det1, det2;
            bool positive_segment;
            int tang_pos_num;
            int view;

            // skip case of equal detectors (as this is a singular LOR)
            if (det_num_a == det_num_b)
              continue;

            positive_segment
                = proj_data_info.get_view_tangential_pos_num_for_det_num_pair(view, tang_pos_num, det_num_a, det_num_b);
            proj_data_info.get_det_num_pair_for_view_tangential_pos_num(det1, det2, view, tang_pos_num);
            if (!check((det_num_a == det1 && det_num_b == det2 && positive_segment)
                       || (det_num_a == det2 && det_num_b == det1 && !positive_segment)))
              {
#  ifdef STIR_OPENMP
                // add a pragma to avoid cerr output being jumbled up if there are any errors
#    pragma omp critical(TESTPROJDATAINFO)
#  endif
                cerr << "Problem at det1 = " << det_num_a << ", det2 = " << det_num_b
                     << "\n  dets -> sino -> dets gives new detector numbers " << det1 << ", " << det2 << endl;
                continue;
              }
            if (!check(view < num_detectors / 2))
              {
#  ifdef STIR_OPENMP
                // add a pragma to avoid cerr output being jumbled up if there are any errors
#    pragma omp critical(TESTPROJDATAINFO)
#  endif
                cerr << "Problem at det1 = " << det_num_a << ", det2 = " << det_num_b << ":\n  view is too big : " << view
                     << endl;
              }
            if (!check(tang_pos_num < num_detectors / 2 && tang_pos_num >= -(num_detectors / 2)))
              {
#  ifdef STIR_OPENMP
                // add a pragma to avoid cerr output being jumbled up if there are any errors
#    pragma omp critical(TESTPROJDATAINFO)
#  endif
                cerr << "Problem at det1 = " << det_num_a << ", det2 = " << det_num_b
                     << ":\n  tang_pos_num is out of range : " << tang_pos_num << endl;
              }
          } // end of detectors_to_sinogram, sinogram_to_detector test

#  ifdef STIR_OPENMP
#    pragma omp parallel for
#  endif
      for (int view = 0; view < num_detectors / 2; ++view)
        for (int tang_pos_num = -(num_detectors / 2) + 1; tang_pos_num < num_detectors / 2; ++tang_pos_num)
          {
            int new_tang_pos_num, new_view;
            bool positive_segment;
            int det_num_a;
            int det_num_b;

            proj_data_info.get_det_num_pair_for_view_tangential_pos_num(det_num_a, det_num_b, view, tang_pos_num);
            positive_segment
                = proj_data_info.get_view_tangential_pos_num_for_det_num_pair(new_view, new_tang_pos_num, det_num_a, det_num_b);

            if (tang_pos_num != new_tang_pos_num || view != new_view || !positive_segment)
              {
#  ifdef STIR_OPENMP
                // add a pragma to avoid cerr output being jumbled up if there are any errors
#    pragma omp critical(TESTPROJDATAINFO)
#  endif
                {
                  cerr << "Problem at view = " << view << ", tang_pos_num = " << tang_pos_num
                       << "\n   sino -> dets -> sino gives new view, tang_pos_num :" << new_view << ", " << new_tang_pos_num
                       << " with detector swapping " << positive_segment << endl;
                }
              }
          } // end of sinogram_to_detector, detectors_to_sinogram test

    } // end of tests that work only without mashing

  if (proj_data_info.get_view_mashing_factor() == 1
      && proj_data_info.get_max_ring_difference(0) == proj_data_info.get_min_ring_difference(0)
      && proj_data_info.get_max_ring_difference(1) == proj_data_info.get_min_ring_difference(1)
      && proj_data_info.get_max_ring_difference(2) == proj_data_info.get_min_ring_difference(2))
    {
      // these tests work only without mashing and axial compression

      cerr << "\n\tTest code for detector,ring -> bin and back conversions.";

      DetectionPositionPair<> det_pos_pair;
      for (det_pos_pair.pos1().axial_coord() = 0; det_pos_pair.pos1().axial_coord() <= 2; det_pos_pair.pos1().axial_coord()++)
        for (det_pos_pair.pos2().axial_coord() = 0; det_pos_pair.pos2().axial_coord() <= 2; det_pos_pair.pos2().axial_coord()++)
#  ifdef STIR_OPENMP
        // insert a parallel for here for testing.
        // we do it at this level to avoid too much overhead for the thread creation, while still having enough jobs to do
        // note: for-loop writing somewhat awkwardly as openmp needs int variables for the loop
#    pragma omp parallel for firstprivate(det_pos_pair)
#  endif
          for (int tangential_coord1 = 0; tangential_coord1 < num_detectors; tangential_coord1++)
            for (det_pos_pair.pos2().tangential_coord() = 0; det_pos_pair.pos2().tangential_coord() < (unsigned)num_detectors;
                 det_pos_pair.pos2().tangential_coord()++)
              for (det_pos_pair.timing_pos() = 0; // currently unsigned so start from 0
                   det_pos_pair.timing_pos() <= (unsigned)proj_data_info.get_max_tof_pos_num();
                   det_pos_pair.timing_pos() += (unsigned)std::max(1, proj_data_info.get_max_tof_pos_num()))
                {

                  // set from for-loop variable
                  det_pos_pair.pos1().tangential_coord() = (unsigned)tangential_coord1;
                  // skip case of equal detector numbers (as this is either a singular LOR)
                  // or an LOR parallel to the scanner axis
                  if (det_pos_pair.pos1().tangential_coord() == det_pos_pair.pos2().tangential_coord())
                    continue;
                  Bin bin(0, 0, 0, 0, 0, 0.0f);
                  DetectionPositionPair<> new_det_pos_pair;
                  const bool there_is_a_bin = proj_data_info.get_bin_for_det_pos_pair(bin, det_pos_pair) == Succeeded::yes;
                  if (there_is_a_bin)
                    proj_data_info.get_det_pos_pair_for_bin(new_det_pos_pair, bin);
#  ifdef STIR_OPENMP
                    // add a pragma to avoid cerr output being jumbled up if there are any errors
#    pragma omp critical(TESTPROJDATAINFO)
#  endif
                  if (!check(there_is_a_bin, "checking if there is a bin for this det_pos_pair")
                      || !check(det_pos_pair == new_det_pos_pair, "checking if we round-trip to the same detection positions"))
                    {
                      cerr << "Problem at det1 = " << det_pos_pair.pos1().tangential_coord()
                           << ", det2 = " << det_pos_pair.pos2().tangential_coord()
                           << ", ring1 = " << det_pos_pair.pos1().axial_coord()
                           << ", ring2 = " << det_pos_pair.pos2().axial_coord() << ", timing_pos = " << det_pos_pair.timing_pos()
                           << endl;
                      if (there_is_a_bin)
                        cerr << "  dets,rings -> bin -> dets,rings, gives new numbers:\n\t"
                             << "det1 = " << new_det_pos_pair.pos1().tangential_coord()

                             << ", det2 = " << new_det_pos_pair.pos2().tangential_coord()
                             << ", ring1 = " << new_det_pos_pair.pos1().axial_coord()
                             << ", ring2 = " << new_det_pos_pair.pos2().axial_coord()
                             << ", timing_pos = " << det_pos_pair.timing_pos() << endl;
                    }

                } // end of get_bin_for_det_pos_pair and vice versa code

      cerr << "\n\tTest code for bin -> detector,ring and back conversions. (This might take a while...)";
      {
        Bin bin(0, 0, 0, 0, 0, 0.0f);
        // set value for comparison later on
        bin.set_bin_value(0.f);
        for (bin.timing_pos_num() = proj_data_info.get_min_tof_pos_num();
             bin.timing_pos_num() <= proj_data_info.get_max_tof_pos_num();
             bin.timing_pos_num() += std::max(1,
                                              (proj_data_info.get_max_tof_pos_num() - proj_data_info.get_min_tof_pos_num())
                                                  / 2)) // take 3 or 1 steps, always going through 0)
          for (bin.segment_num() = max(-5, proj_data_info.get_min_segment_num());
               bin.segment_num() <= min(5, proj_data_info.get_max_segment_num());
               ++bin.segment_num())
            for (bin.axial_pos_num() = proj_data_info.get_min_axial_pos_num(bin.segment_num());
                 bin.axial_pos_num() <= proj_data_info.get_max_axial_pos_num(bin.segment_num());
                 ++bin.axial_pos_num())
#  ifdef STIR_OPENMP
            // insert a parallel for here for testing.
            // we do it at this level to avoid too much overhead for the thread creation, while still having enough jobs to do
            // Note that the omp construct needs an int loop variable
#    pragma omp parallel for firstprivate(bin)
#  endif
              for (int tangential_pos_num = -(num_detectors / 2) + 1; tangential_pos_num < num_detectors / 2;
                   ++tangential_pos_num)
                for (bin.view_num() = 0; bin.view_num() < num_detectors / 2; ++bin.view_num())
                  {
                    // set from for-loop variable
                    bin.tangential_pos_num() = tangential_pos_num;
                    Bin new_bin(0, 0, 0, 0, 0, 0.0f);
                    // set value for comparison with bin
                    new_bin.set_bin_value(0);
                    DetectionPositionPair<> det_pos_pair;
                    proj_data_info.get_det_pos_pair_for_bin(det_pos_pair, bin);

                    const bool there_is_a_bin = proj_data_info.get_bin_for_det_pos_pair(new_bin, det_pos_pair) == Succeeded::yes;
#  ifdef STIR_OPENMP
                    // add a pragma to avoid cerr output being jumbled up if there are any errors
#    pragma omp critical(TESTPROJDATAINFO)
#  endif
                    if (!check(there_is_a_bin, "checking if there is a bin for this det_pos_pair")
                        || !check(bin == new_bin, "checking if we round-trip to the same bin"))
                      {
                        cerr << "Problem at  segment = " << bin.segment_num() << ", axial pos " << bin.axial_pos_num()
                             << ", view = " << bin.view_num() << ", tangential_pos_num = " << bin.tangential_pos_num()
                             << ", timing pos num = " << bin.timing_pos_num() << "\n";
                        if (there_is_a_bin)
                          cerr << "  bin -> dets -> bin, gives new numbers:\n\t"
                               << "segment = " << new_bin.segment_num() << ", axial pos " << new_bin.axial_pos_num()
                               << ", view = " << new_bin.view_num() << ", tangential_pos_num = " << new_bin.tangential_pos_num()
                               << ", timing pos num = " << new_bin.timing_pos_num() << endl;
                      }

                  } // end of get_det_pos_pair_for_bin and back code
      }
    } // end of tests which require no mashing nor axial compression

  {
    cerr << "\n\tTest code for bins <-> detectors routines that work with any mashing and axial compression";

    Bin bin;
    // set value for comparison later on
    bin.set_bin_value(0);
    // parallel loop for testing
    // Note that the omp construct cannot handle bin.segment_num() etc as loop variable, making this ugly
#  ifdef STIR_OPENMP
#    if _OPENMP < 201107
#      pragma omp parallel for schedule(dynamic) firstprivate(bin)
#    else
#      pragma omp parallel for collapse(2) firstprivate(bin)
#    endif
#  endif
    for (int segment_num = proj_data_info.get_min_segment_num(); segment_num <= proj_data_info.get_max_segment_num();
         ++segment_num)
      for (int view_num = proj_data_info.get_min_view_num(); view_num <= proj_data_info.get_max_view_num(); ++view_num)
        for (int axial_pos_num = proj_data_info.get_min_axial_pos_num(segment_num);
             axial_pos_num <= proj_data_info.get_max_axial_pos_num(segment_num);
             ++axial_pos_num)
          for (int tangential_pos_num = proj_data_info.get_min_tangential_pos_num();
               tangential_pos_num <= proj_data_info.get_max_tangential_pos_num();
               ++tangential_pos_num)
            for (bin.timing_pos_num() = proj_data_info.get_min_tof_pos_num();
                 bin.timing_pos_num() <= proj_data_info.get_max_tof_pos_num();
                 bin.timing_pos_num() += std::max(1,
                                                  (proj_data_info.get_max_tof_pos_num() - proj_data_info.get_min_tof_pos_num())
                                                      / 2)) // take 3 or 1 steps, always going through 0)
              {
                bin.segment_num() = segment_num;
                bin.axial_pos_num() = axial_pos_num;
                bin.view_num() = view_num;
                bin.tangential_pos_num() = tangential_pos_num;
                std::vector<DetectionPositionPair<>> det_pos_pairs;
                proj_data_info.get_all_det_pos_pairs_for_bin(det_pos_pairs, bin, false); // include TOF
                Bin new_bin;
                // set value for comparison with bin
                new_bin.set_bin_value(0);
                for (std::vector<DetectionPositionPair<>>::const_iterator det_pos_pair_iter = det_pos_pairs.begin();
                     det_pos_pair_iter != det_pos_pairs.end();
                     ++det_pos_pair_iter)
                  {
                    const bool there_is_a_bin
                        = proj_data_info.get_bin_for_det_pos_pair(new_bin, *det_pos_pair_iter) == Succeeded::yes;
                    if (!check(there_is_a_bin, "checking if there is a bin for this det_pos_pair")
                        || !check(bin == new_bin, "checking if we round-trip to the same bin"))
                      {
#  ifdef STIR_OPENMP
                        // add a pragma to avoid cerr output being jumbled up if there are any errors
#    pragma omp critical(TESTPROJDATAINFO)
#  endif
                        {
                          cerr << "Problem at  segment = " << bin.segment_num() << ", axial pos " << bin.axial_pos_num()
                               << ", view = " << bin.view_num() << ", tangential_pos_num = " << bin.tangential_pos_num() << "\n";
                          if (there_is_a_bin)
                            cerr << "  bin -> dets -> bin, gives new numbers:\n\t"
                                 << "segment = " << new_bin.segment_num() << ", axial pos " << new_bin.axial_pos_num()
                                 << ", view = " << new_bin.view_num() << ", tangential_pos_num = " << new_bin.tangential_pos_num()
                                 << ", timing_pos - " << new_bin.timing_pos_num() << endl;
                        }
                      }
                  } // end of iteration of det_pos_pairs
              }     // end of loop over all bins
  }                 // end of get_all_det_pairs_for_bin and back code
#endif              // TEST_ONLY_GET_BIN

  {
    cerr << endl;
    cerr << "\tTesting find scanner coordinates given cartesian and vice versa." << endl;
    {
      const int num_detectors_per_ring = proj_data_info.get_scanner_ptr()->get_num_detectors_per_ring();
      const int num_rings = proj_data_info.get_scanner_ptr()->get_num_rings();

#ifdef STIR_OPENMP
#  pragma omp parallel for schedule(dynamic)
#endif
      for (int Ring_A = 0; Ring_A < num_rings; Ring_A += num_rings / 3)
        for (int Ring_B = 0; Ring_B < num_rings; Ring_B += num_rings / 3)
          for (int det1 = 0; det1 < num_detectors_per_ring; ++det1)
            for (int det2 = 0; det2 < num_detectors_per_ring; ++det2)
              {
                if (det1 == det2)
                  continue;
                CartesianCoordinate3D<float> coord_1;
                CartesianCoordinate3D<float> coord_2;

                proj_data_info.find_cartesian_coordinates_given_scanner_coordinates(
                    coord_1, coord_2, Ring_A, Ring_B, det1, det2, 1); // use timing_pos_num>=0 as pre-TOF test

                const CartesianCoordinate3D<float> coord_1_new = coord_1 + (coord_2 - coord_1) * 5;
                const CartesianCoordinate3D<float> coord_2_new = coord_1 + (coord_2 - coord_1) * 2;

                int det1_f, det2_f, ring1_f, ring2_f;

                check(proj_data_info.find_scanner_coordinates_given_cartesian_coordinates(
                          det1_f, det2_f, ring1_f, ring2_f, coord_1_new, coord_2_new)
                      == Succeeded::yes);
                if (det1_f == det1 && Ring_A == ring1_f)
                  {
                    check_if_equal(det1_f, det1, "test on det1");
                    check_if_equal(Ring_A, ring1_f, "test on ring1");
                    check_if_equal(det2_f, det2, "test on det2");
                    check_if_equal(Ring_B, ring2_f, "test on ring1");
                  }
                else
                  {
                    check_if_equal(det2_f, det1, "test on det1");
                    check_if_equal(Ring_B, ring1_f, "test on ring1");
                    check_if_equal(det1_f, det2, "test on det2");
                    check_if_equal(Ring_A, ring2_f, "test on ring1");
                  }
              }
    }
  }
}

END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int
main()
{
  set_default_num_threads();

#ifndef STIR_TOF_DEBUG // disable for speed of testing
  {
    ProjDataInfoCylindricalArcCorrTests tests;
    tests.run_tests();
    if (!tests.is_everything_ok())
      return tests.main_return_value();
  }
#endif
  {
    ProjDataInfoCylindricalNoArcCorrTests tests1;
    tests1.run_tests();
    return tests1.main_return_value();
  }
}
