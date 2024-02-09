/*!

  \file
  \ingroup recontest

  \brief Test program for detection position map using stir::ProjDataInfoBlockOnCylindrical

  \author Daniel Deidda

*/
/*  Copyright (C) 2021, National Physical Laboratory
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/

#include "stir/info.h"
#include "stir/make_array.h"
#include "stir/ProjDataInMemory.h"
#include "stir/DiscretisedDensity.h"
#include "stir/ProjDataInterfile.h"
#include "stir/recon_buildblock/ProjMatrixElemsForOneBin.h"
#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#include "stir/ExamInfo.h"
#include "stir/LORCoordinates.h"
#include "stir/ProjDataInfo.h"
#include "stir/ProjDataInfoBlocksOnCylindricalNoArcCorr.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#include "stir/Sinogram.h"
#include "stir/Viewgram.h"
#include "stir/Succeeded.h"
#include "stir/RunTests.h"
#include "stir/Scanner.h"
#include "stir/copy_fill.h"
#include "stir/IndexRange3D.h"
#include "stir/CPUTimer.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/recon_buildblock/ForwardProjectorByBin.h"
#include "stir/recon_buildblock/ForwardProjectorByBinUsingProjMatrixByBin.h"
#include "stir/IO/write_to_file.h"
// #include "stir/Shape/Shape3D.h"

#include "stir/Shape/Shape3DWithOrientation.h"
#include "stir/Shape/Ellipsoid.h"
#include "stir/Shape/Box3D.h"

START_NAMESPACE_STIR
/*!
  \ingroup test
  \brief Test class for Blocks
*/
class DetectionPosMapTests : public RunTests
{
public:
  void run_tests() override;
  float calculate_angle_within_half_bucket(const shared_ptr<Scanner> scanner_ptr,
                                           const shared_ptr<ProjDataInfoBlocksOnCylindricalNoArcCorr> proj_data_info_ptr);

private:
  void run_coordinate_test_for_flat_first_bucket();
};

float
DetectionPosMapTests::calculate_angle_within_half_bucket(
    const shared_ptr<Scanner> scanner_ptr, const shared_ptr<ProjDataInfoBlocksOnCylindricalNoArcCorr> proj_data_info_ptr)
{
  Bin bin;
  LORInAxialAndNoArcCorrSinogramCoordinates<float> lorB;
  float csi;
  float C_spacing = scanner_ptr->get_transaxial_crystal_spacing();
  float csi_crystal = std::atan((C_spacing) / scanner_ptr->get_effective_ring_radius());
  //    float bucket_spacing=scanner_ptr->get_transaxial_block_spacing()*C_spacing;
  //    float blocks_gap=scanner_ptr->get_transaxial_block_spacing()
  //            -scanner_ptr->get_num_transaxial_crystals_per_block()*C_spacing;
  //    float csi_gap=std::atan((blocks_gap)/scanner_ptr->get_effective_ring_radius());

  //    get angle within half bucket
  for (int view = 0; view <= scanner_ptr->get_max_num_views(); view++)
    {
      int bucket_num
          = view / (scanner_ptr->get_num_transaxial_crystals_per_block() * scanner_ptr->get_num_transaxial_blocks_per_bucket());
      if (bucket_num > 0)
        break;

      bin.segment_num() = 0;
      bin.axial_pos_num() = 0;
      bin.view_num() = view;
      bin.tangential_pos_num() = 0;

      proj_data_info_ptr->get_LOR(lorB, bin);
      csi = lorB.phi();
    }
  return (csi + csi_crystal) / 2;
}

/*!
  The following test checks that the y position of the detectors in buckets that are parallel to the x axis are the same.
  The calculation of csi is only valid for the scanner defined in stir::Scanner, if we modify the number of blocks per bucket
  csi will be affected. However this does not happen when csi is calculated in the same way we do in the crystal map.
*/
void
DetectionPosMapTests::run_coordinate_test_for_flat_first_bucket()
{
  CPUTimer timer;
  auto scannerBlocks_ptr = std::make_shared<Scanner>(Scanner::SAFIRDualRingPrototype);

  scannerBlocks_ptr->set_scanner_geometry("BlocksOnCylindrical");
  scannerBlocks_ptr->set_num_transaxial_blocks_per_bucket(1);
  scannerBlocks_ptr->set_up();

  VectorWithOffset<int> num_axial_pos_per_segment(scannerBlocks_ptr->get_num_rings() * 2 - 1);
  VectorWithOffset<int> min_ring_diff_v(scannerBlocks_ptr->get_num_rings() * 2 - 1);
  VectorWithOffset<int> max_ring_diff_v(scannerBlocks_ptr->get_num_rings() * 2 - 1);

  for (int i = 0; i < 2 * scannerBlocks_ptr->get_num_rings() - 1; i++)
    {
      min_ring_diff_v[i] = -scannerBlocks_ptr->get_num_rings() + 1 + i;
      max_ring_diff_v[i] = -scannerBlocks_ptr->get_num_rings() + 1 + i;
      if (i < scannerBlocks_ptr->get_num_rings())
        num_axial_pos_per_segment[i] = i + 1;
      else
        num_axial_pos_per_segment[i] = 2 * scannerBlocks_ptr->get_num_rings() - i - 1;
    }

  auto proj_data_info_blocks_ptr
      = std::make_shared<ProjDataInfoBlocksOnCylindricalNoArcCorr>(scannerBlocks_ptr,
                                                                   num_axial_pos_per_segment,
                                                                   min_ring_diff_v,
                                                                   max_ring_diff_v,
                                                                   scannerBlocks_ptr->get_max_num_views(),
                                                                   scannerBlocks_ptr->get_max_num_non_arccorrected_bins());

  Bin bin, bin0 = Bin(0, 0, 0, 0);
  CartesianCoordinate3D<float> b1, b2, b01, b02;

  //    estimate the angle covered by half bucket, csi
  float csi;
  csi = calculate_angle_within_half_bucket(scannerBlocks_ptr, proj_data_info_blocks_ptr);

  auto scannerBlocks_firstFlat_ptr = std::make_shared<Scanner>(Scanner::SAFIRDualRingPrototype);
  scannerBlocks_firstFlat_ptr->set_scanner_geometry("BlocksOnCylindrical");
  scannerBlocks_firstFlat_ptr->set_num_transaxial_blocks_per_bucket(1);
  scannerBlocks_firstFlat_ptr->set_intrinsic_azimuthal_tilt(-csi);
  scannerBlocks_firstFlat_ptr->set_up();

  auto proj_data_info_blocks_firstFlat_ptr = std::make_shared<ProjDataInfoBlocksOnCylindricalNoArcCorr>(
      scannerBlocks_firstFlat_ptr,
      num_axial_pos_per_segment,
      min_ring_diff_v,
      max_ring_diff_v,
      scannerBlocks_firstFlat_ptr->get_max_num_views(),
      scannerBlocks_firstFlat_ptr->get_max_num_non_arccorrected_bins());
  timer.reset();
  timer.start();

  for (int view = 0; view <= proj_data_info_blocks_firstFlat_ptr->get_max_view_num(); view++)
    {
      int bucket_num = view
                       / (scannerBlocks_firstFlat_ptr->get_num_transaxial_crystals_per_block()
                          * scannerBlocks_firstFlat_ptr->get_num_transaxial_blocks_per_bucket());
      if (bucket_num > 0)
        break;

      bin.segment_num() = 0;
      bin.axial_pos_num() = 0;
      bin.view_num() = view;
      bin.tangential_pos_num() = 0;

      //                check cartesian coordinates of detectors
      proj_data_info_blocks_firstFlat_ptr->find_cartesian_coordinates_of_detection(b1, b2, bin);
      proj_data_info_blocks_firstFlat_ptr->find_cartesian_coordinates_of_detection(b01, b02, bin0);

      check_if_equal(b1.y(), b01.y(), " checking cartesian coordinate y1 are the same on a flat bucket");
      check_if_equal(b2.y(), b02.y(), " checking cartesian coordinate y2 are the same on a flat bucket");
      check_if_equal(b1.y(), -b2.y(), " checking cartesian coordinate y1 and y2 are of opposite sign on opposite flat buckets");
    }
  timer.stop();
  std::cerr << "-- CPU Time " << timer.value() << '\n';
}

void
DetectionPosMapTests::run_tests()
{

  std::cerr << "-------- Testing DetectorCoordinateMap --------\n";
  run_coordinate_test_for_flat_first_bucket();
}
END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int
main()
{
  DetectionPosMapTests tests;
  tests.run_tests();
  return tests.main_return_value();
}
