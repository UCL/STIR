//
//
/*!

  \file
  \ingroup test

  \brief Test program for ML_norm.h functionality

  \author Kris Thielemans
  \author daniel deidda
*/
/*
    Copyright (C) 2021, University College London
    Copyright (C) 2022, National Physical Laboratory
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/ProjDataInMemory.h"
#include "stir/RunTests.h"
#include "stir/Scanner.h"
#include "stir/Bin.h"
#include "stir/ML_norm.h"
#include "stir/IndexRange2D.h"
#include "stir/numerics/norm.h"
#include "stir/num_threads.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <math.h>

START_NAMESPACE_STIR

/*!
  \ingroup test
  \brief Test class for ML_norm.h functions
*/
class ML_normTests : public RunTests
{
public:
  void run_tests() override;

protected:
  template <class TProjDataInfo>
  void test_proj_data_info(shared_ptr<TProjDataInfo> proj_data_info_sptr);
};

void
ML_normTests::run_tests()
{
  {
    std::cerr << "\n-------- Testing ECAT 953 --------\n";
    shared_ptr<Scanner> scanner_sptr(new Scanner(Scanner::E953));
    shared_ptr<ProjDataInfo> proj_data_info_sptr(
        ProjDataInfo::construct_proj_data_info(scanner_sptr,
                                               /*span*/ 1,
                                               scanner_sptr->get_num_rings() - 1,
                                               /*views*/ scanner_sptr->get_num_detectors_per_ring() / 2,
                                               /*tang_pos*/ 64,
                                               /*arc_corrected*/ false));
    test_proj_data_info(dynamic_pointer_cast<ProjDataInfoCylindricalNoArcCorr>(proj_data_info_sptr));
  }
  {
    std::cerr << "\n-------- Testing ECAT E1080 (with gaps) --------\n";
    shared_ptr<Scanner> scanner_sptr(new Scanner(Scanner::E1080));
    shared_ptr<ProjDataInfo> proj_data_info_sptr(
        ProjDataInfo::construct_proj_data_info(scanner_sptr,
                                               /*span*/ 1,
                                               scanner_sptr->get_num_rings() - 1,
                                               /*views*/ scanner_sptr->get_num_detectors_per_ring() / 2,
                                               /*tang_pos*/ 64,
                                               /*arc_corrected*/ false));
    test_proj_data_info(dynamic_pointer_cast<ProjDataInfoCylindricalNoArcCorr>(proj_data_info_sptr));
  }
  {
    std::cerr << "\n-------- Testing Block Scanner SAFIR --------\n";
    shared_ptr<Scanner> scanner_sptr(new Scanner(Scanner::SAFIRDualRingPrototype));
    scanner_sptr->set_scanner_geometry("BlocksOnCylindrical");
    scanner_sptr->set_up();
    shared_ptr<ProjDataInfo> proj_data_info_sptr(
        ProjDataInfo::construct_proj_data_info(scanner_sptr,
                                               /*span*/ 1,
                                               scanner_sptr->get_num_rings() - 1,
                                               /*views*/ scanner_sptr->get_num_detectors_per_ring() / 2,
                                               /*tang_pos*/ 64,
                                               /*arc_corrected*/ false));
    test_proj_data_info(dynamic_pointer_cast<ProjDataInfoBlocksOnCylindricalNoArcCorr>(proj_data_info_sptr));
  }
}

template <class TProjDataInfo>
void
ML_normTests::test_proj_data_info(shared_ptr<TProjDataInfo> proj_data_info_sptr)
{
  if (!check(proj_data_info_sptr != nullptr, "check type of proj_data_info"))
    return;
  // const int num_detectors = proj_data_info.get_scanner_ptr()->get_num_detectors_per_ring();
  auto exam_info_sptr = std::make_shared<ExamInfo>();
  ProjDataInMemory proj_data(exam_info_sptr, proj_data_info_sptr);
  proj_data.fill(1.F);

  const int num_virtual_axial_crystals_per_block
      = proj_data_info_sptr->get_scanner_sptr()->get_num_virtual_axial_crystals_per_block();
  const int num_virtual_transaxial_crystals_per_block
      = proj_data_info_sptr->get_scanner_sptr()->get_num_virtual_transaxial_crystals_per_block();
  const int num_transaxial_crystals_per_block = proj_data_info_sptr->get_scanner_sptr()->get_num_transaxial_crystals_per_block();
  const int num_axial_crystals_per_block = proj_data_info_sptr->get_scanner_sptr()->get_num_axial_crystals_per_block();
  const int num_physical_transaxial_crystals_per_block
      = num_transaxial_crystals_per_block - num_virtual_transaxial_crystals_per_block;
  // const int num_physical_axial_crystals_per_block = num_axial_crystals_per_block - num_virtual_axial_crystals_per_block;
  const int num_physical_rings
      = proj_data_info_sptr->get_scanner_sptr()->get_num_rings()
        - (proj_data_info_sptr->get_scanner_sptr()->get_num_axial_blocks() - 1) * num_virtual_axial_crystals_per_block;
  const int num_physical_detectors_per_ring
      = proj_data_info_sptr->get_scanner_sptr()->get_num_detectors_per_ring()
        - proj_data_info_sptr->get_scanner_sptr()->get_num_transaxial_blocks() * num_virtual_transaxial_crystals_per_block;

  FanProjData fan_data;
  make_fan_data_remove_gaps(fan_data, proj_data);
  {
    ProjDataInMemory proj_data2(proj_data);
    proj_data2.fill(0.F);
    set_fan_data_add_gaps(proj_data2, fan_data, /*gap_value=*/1.F);
    {
      // test if the same after round-trip if we fill the gap
      // proj_data2 -= proj_data;
      proj_data2.sapyb(1.F, proj_data, -1.F);
      check_if_zero(norm_squared(proj_data2.begin(), proj_data2.end())
                        / square(static_cast<double>(proj_data_info_sptr->size_all())),
                    "projdata <-> fandata with gap filled");
    }

    set_fan_data_add_gaps(proj_data2, fan_data, /*gap_value=*/0.F);
    {
      // test round-trip if we do not fill the gap
      {
        {
          // test view 0, should have a values 1,1,1,...,0 (starting from the middle)
          const int view_num = 0;
          const int axial_pos_num = 0;
          for (int seg_num = -1; seg_num <= 1; ++seg_num)
            for (int tangential_pos_num = 0; tangential_pos_num <= proj_data2.get_max_tangential_pos_num(); ++tangential_pos_num)
              {
                Bin bin(seg_num, view_num, axial_pos_num, tangential_pos_num);
                proj_data2.get_bin_value(bin);
                const float value = bin.get_bin_value();
                proj_data.get_bin_value(bin);
                const float org_value = bin.get_bin_value();
                const bool in_gap
                    = (tangential_pos_num % num_transaxial_crystals_per_block) >= num_physical_transaxial_crystals_per_block;
                if (in_gap)
                  check_if_zero(value, "projdata <-> fandata with gap zero: in gap (sino)");
                else
                  check_if_equal(value, org_value, "projdata <-> fandata with gap zero: not in gap (sino)");
              }
        }
        if (num_virtual_axial_crystals_per_block > 0)
          {
            // test a sinogram in segment 0 in the gap
            const Sinogram<float> sino = proj_data2.get_sinogram(num_axial_crystals_per_block - 1, /* segment*/ 0);
            if (!check_if_zero(sino.find_max(), "projdata <-> fandata with gap zero: in gap (axial)"))
              {
                const std::string filename = "test_gaps_" + proj_data_info_sptr->get_scanner_sptr()->get_name() + ".hs";
                std::cerr << "writing filled data with gaps to file for debugging " << filename << "\n";
                proj_data2.write_to_file(filename);
              }
          }
      }

      // test make_fan_sum_data if there are no gaps. In this case, all fan sums should be equal
      if (num_virtual_transaxial_crystals_per_block == 0 && num_virtual_axial_crystals_per_block == 0)
        {
          Array<2, float> data_fan_sums(IndexRange2D(num_physical_rings, num_physical_detectors_per_ring));
          make_fan_sum_data(data_fan_sums, fan_data);
          check_if_equal(data_fan_sums.find_min(), data_fan_sums.find_max(), "make_fan_sum_data (no gaps)");
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

  {
    ML_normTests tests;
    tests.run_tests();
    if (!tests.is_everything_ok())
      return tests.main_return_value();
  }
  return EXIT_SUCCESS;
}
