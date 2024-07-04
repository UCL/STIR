//
//
/*
  Copyright 2023, Positrigo AG, Zurich
  Copyright 2024, University College London
  This file is part of STIR.

  SPDX-License-Identifier: Apache-2.0

  See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup test

  \brief Tests for stir::ProjData interpolation as used by the scatter estimation.

  \author Markus Jehl
*/

#ifndef NDEBUG
// set to high level of debugging
#  ifdef _DEBUG
#    undef _DEBUG
#  endif
#  define _DEBUG 2
#endif

#include "stir/ProjDataInfo.h"
#include "stir/ExamInfo.h"
#include "stir/ProjDataInMemory.h"
#include "stir/Succeeded.h"
#include "stir/IO/write_data.h"
#include "stir/IO/read_data.h"
#include "stir/IO/write_to_file.h"
#include "stir/numerics/BSplines.h"
#include "stir/interpolate_projdata.h"
#include "stir/inverse_SSRB.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/Shape/EllipsoidalCylinder.h"
#include "stir/Shape/Box3D.h"
#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#include "stir/recon_buildblock/ForwardProjectorByBinUsingProjMatrixByBin.h"

#include "stir/RunTests.h"

START_NAMESPACE_STIR

class InterpolationTests : public RunTests
{
public:
  void run_tests() override;

private:
  void scatter_interpolation_test_blocks();
  void scatter_interpolation_test_cyl();
  void scatter_interpolation_test_blocks_asymmetric();
  void scatter_interpolation_test_cyl_asymmetric();
  void scatter_interpolation_test_blocks_downsampled();

  void check_symmetry(const SegmentBySinogram<float>& segment);
  void compare_segment(const SegmentBySinogram<float>& segment1, const SegmentBySinogram<float>& segment2, float maxDiff);
  void
  compare_segment_shape(const SegmentBySinogram<float>& shape_segment, const SegmentBySinogram<float>& test_segment, int erosion);
};

void
InterpolationTests::check_symmetry(const SegmentBySinogram<float>& segment)
{
  // compare lower half of slices with upper half - image should be axially symmetric
  auto maxAbsDifference = 0.0;
  auto sumAbsValues = 0.0;
  auto summedEntries = 0.0;
  auto increasing_index = segment.get_min_axial_pos_num();
  auto decreasing_index = segment.get_max_axial_pos_num();
  while (increasing_index < decreasing_index)
    {
      for (auto view = segment.get_min_view_num(); view <= segment.get_max_view_num(); view++)
        {
          for (auto tang = segment.get_min_tangential_pos_num(); tang <= segment.get_max_tangential_pos_num(); tang++)
            {
              auto voxel1 = std::abs(segment[increasing_index][view][tang]);
              auto voxel2 = std::abs(segment[decreasing_index][view][tang]);
              if (std::abs(voxel1 - voxel2) > maxAbsDifference)
                maxAbsDifference = std::abs(voxel1 - voxel2);
              if (voxel1 > 0)
                {
                  sumAbsValues += voxel1;
                  summedEntries++;
                }
              if (voxel2 > 0)
                {
                  sumAbsValues += voxel2;
                  summedEntries++;
                }
            }
        }
      increasing_index++;
      decreasing_index--;
    }
  // if the largest symmetry error is larger than 0.01% of the mean absolute value, then there is something wrong
  check_if_less(maxAbsDifference,
                0.0001 * sumAbsValues / summedEntries,
                "symmetry errors larger than 0.01\% of absolute values in axial direction");

  // compare the first half of the views with the second half - even for the BlocksOnCylindrical scanner they should be identical
  maxAbsDifference = 0.0;
  sumAbsValues = 0.0;
  summedEntries = 0.0;
  for (auto view = 0; view < segment.get_num_views() / 2; view++)
    {
      for (auto axial = segment.get_min_axial_pos_num(); axial <= segment.get_max_axial_pos_num(); axial++)
        {
          for (auto tang = segment.get_min_tangential_pos_num(); tang <= segment.get_max_tangential_pos_num(); tang++)
            {
              auto voxel1 = segment[axial][view][tang];
              auto voxel2 = segment[axial][view + segment.get_num_views() / 2][tang];
              if (std::abs(voxel1 - voxel2) > maxAbsDifference)
                maxAbsDifference = std::abs(voxel1 - voxel2);
              if (voxel1 > 0)
                {
                  sumAbsValues += voxel1;
                  summedEntries++;
                }
              if (voxel2 > 0)
                {
                  sumAbsValues += voxel2;
                  summedEntries++;
                }
            }
        }
    }
  // if the largest symmetry error is larger than 0.1% of the mean absolute value, then there is something wrong
  // TODO: this tolerance can be tightened to 0.01% if https://github.com/UCL/STIR/issues/1176 is resolved
  check_if_less(maxAbsDifference,
                0.001 * sumAbsValues / summedEntries,
                "symmetry errors larger than 0.1\% of absolute values across views");
}

void
InterpolationTests::compare_segment(const SegmentBySinogram<float>& segment1,
                                    const SegmentBySinogram<float>& segment2,
                                    float maxDiff)
{
  // compute difference and compare against empirically found value from visually validated sinograms
  auto sumAbsDifference = 0.0;
  for (auto axial = segment1.get_min_axial_pos_num(); axial <= segment1.get_max_axial_pos_num(); axial++)
    {
      for (auto view = segment1.get_min_view_num(); view <= segment1.get_max_view_num(); view++)
        {
          for (auto tang = segment1.get_min_tangential_pos_num(); tang <= segment1.get_max_tangential_pos_num(); tang++)
            {
              sumAbsDifference += std::abs(segment1[axial][view][tang] - segment2[axial][view][tang]);
            }
        }
    }

  // confirm that the difference is smaller than an empirically found value
  check_if_less(sumAbsDifference, maxDiff, "difference between segments is larger than expected");
}

void
InterpolationTests::compare_segment_shape(const SegmentBySinogram<float>& shape_segment,
                                          const SegmentBySinogram<float>& test_segment,
                                          int erosion)
{
  auto maxTestValue = test_segment.find_max();
  // compute difference and compare against empirically found value from visually validated sinograms
  auto sumVoxelsOutsideMask = 0U;
  for (auto axial = test_segment.get_min_axial_pos_num(); axial <= test_segment.get_max_axial_pos_num(); axial++)
    {
      for (auto view = test_segment.get_min_view_num(); view <= test_segment.get_max_view_num(); view++)
        {
          for (auto tang = test_segment.get_min_tangential_pos_num(); tang <= test_segment.get_max_tangential_pos_num(); tang++)
            {
              if (test_segment[axial][view][tang] < 0.1 * maxTestValue)
                continue;

              // now go through the erosion neighbourhood of the voxel to see if it is near a non-zero voxel
              bool isNearNonZero = false;
              for (auto axialShape = std::max(axial - erosion, test_segment.get_min_axial_pos_num());
                   axialShape <= std::min(axial + erosion, test_segment.get_max_axial_pos_num());
                   axialShape++)
                {
                  for (auto viewShape = std::max(view - erosion, test_segment.get_min_view_num());
                       viewShape <= std::min(view + erosion, test_segment.get_max_view_num());
                       viewShape++)
                    {
                      for (auto tangShape = std::max(tang - erosion, test_segment.get_min_tangential_pos_num());
                           tangShape <= std::min(tang + erosion, test_segment.get_max_tangential_pos_num());
                           tangShape++)
                        {
                          if (shape_segment[axialShape][viewShape][tangShape] > 0)
                            isNearNonZero = true;
                        }
                    }
                }
              if (isNearNonZero == false)
                sumVoxelsOutsideMask++;
            }
        }
    }

  // confirm that the difference is smaller than an empirically found value
  check_if_equal(sumVoxelsOutsideMask, 0U, "there were non-zero voxels outside the masked area");
}

static void
make_symmetric_object(VoxelsOnCartesianGrid<float>& emission_map)
{
  const float z_voxel_size = emission_map.get_grid_spacing()[1];
  const float z_centre = (emission_map.get_min_z() + emission_map.get_max_z()) / 2.F * z_voxel_size;
  // choose a length that isn't exactly equal to a number of planes (or half), as that
  // is sensitive to rounding error
  auto cylinder = EllipsoidalCylinder(z_voxel_size * 4.5, 80, 80, CartesianCoordinate3D<float>(z_centre, 0, 0));
  cylinder.construct_volume(emission_map, CartesianCoordinate3D<int>(1, 1, 1));
}

void
InterpolationTests::scatter_interpolation_test_blocks()
{
  info("Performing symmetric interpolation test for BlocksOnCylindrical scanner");
  auto time_frame_def = TimeFrameDefinitions();
  time_frame_def.set_num_time_frames(1);
  time_frame_def.set_time_frame(1, 0, 1e9);
  auto exam_info = ExamInfo();
  exam_info.set_high_energy_thres(650);
  exam_info.set_low_energy_thres(425);
  exam_info.set_time_frame_definitions(time_frame_def);

  // define the original scanner and a downsampled one, as it would be used for scatter simulation
  auto scanner = Scanner(Scanner::User_defined_scanner,
                         "Some_symmetric_scanner",
                         192,
                         30,
                         150,
                         150,
                         127,
                         4.3,
                         4.13793, // total scanner length of 120mm divided by (rings - 1) to get the spacing
                         2.0,
                         -0.38956 /* 0.0 */,
                         5,
                         4,
                         6,
                         6,
                         1,
                         1,
                         1,
                         0.17,
                         511,
                         -1,
                         01.F,
                         -1.F,
                         "BlocksOnCylindrical",
                         4.13793, // total scanner length of 120mm divided by (rings - 1) to get the spacing
                         4.0,
                         24.83, // ring spacing multiplied by number of crystals per block
                         24.0);
  auto downsampled_scanner = Scanner(Scanner::User_defined_scanner,
                                     "Some_symmetric_scanner",
                                     192,
                                     6,
                                     150,
                                     150,
                                     127,
                                     4.3,
                                     24.0, // total scanner length of 120mm divided by (rings - 1) to get the spacing
                                     2.0,
                                     -0.38956 /* 0.0 */,
                                     1,
                                     4,
                                     6,
                                     6,
                                     1,
                                     1,
                                     1,
                                     0.17,
                                     511,
                                     -1,
                                     01.F,
                                     -1.F,
                                     "BlocksOnCylindrical",
                                     24.0, // total scanner length of 120mm divided by (rings - 1) to get the spacing
                                     4.0,
                                     144.0, // ring spacing multiplied by number of crystals per block
                                     24.0);

  auto proj_data_info = shared_ptr<ProjDataInfo>(
      std::move(ProjDataInfo::construct_proj_data_info(std::make_shared<Scanner>(scanner), 1, 29, 96, 150, false)));
  auto downsampled_proj_data_info = shared_ptr<ProjDataInfo>(
      std::move(ProjDataInfo::construct_proj_data_info(std::make_shared<Scanner>(downsampled_scanner), 1, 0, 96, 150, false)));

  auto proj_data = ProjDataInMemory(std::make_shared<ExamInfo>(exam_info), proj_data_info);
  auto downsampled_proj_data = ProjDataInMemory(std::make_shared<ExamInfo>(exam_info), downsampled_proj_data_info);

  // define a cylinder precisely in the middle of the FOV, such that symmetry can be used for validation
  auto emission_map = VoxelsOnCartesianGrid<float>(*downsampled_proj_data_info, 1);
  make_symmetric_object(emission_map);
  write_to_file("downsampled_cylinder_map", emission_map);

  // project the cylinder onto the downsampled scanner proj data
  auto pm = ProjMatrixByBinUsingRayTracing();
  pm.set_use_actual_detector_boundaries(true);
  pm.enable_cache(false);
  auto forw_proj = ForwardProjectorByBinUsingProjMatrixByBin(std::make_shared<ProjMatrixByBinUsingRayTracing>(pm));
  forw_proj.set_up(downsampled_proj_data_info, std::make_shared<VoxelsOnCartesianGrid<float>>(emission_map));
  auto downsampled_model_sino = ProjDataInMemory(downsampled_proj_data);
  downsampled_model_sino.fill(0);
  forw_proj.forward_project(downsampled_model_sino, emission_map);

  // interpolate the downsampled proj data to the original scanner size and fill in oblique sinograms
  auto interpolated_direct_proj_data = ProjDataInMemory(proj_data);
  interpolate_projdata(interpolated_direct_proj_data, downsampled_model_sino, BSpline::linear, false);
  auto interpolated_proj_data = ProjDataInMemory(proj_data);
  inverse_SSRB(interpolated_proj_data, interpolated_direct_proj_data);

  // write the proj data to file
  downsampled_model_sino.write_to_file("downsampled_sino.hs");
  interpolated_proj_data.write_to_file("interpolated_sino.hs");

  // use symmetry to check that there are no significant errors in the interpolation
  check_symmetry(interpolated_proj_data.get_segment_by_sinogram(0));
}

void
InterpolationTests::scatter_interpolation_test_cyl()
{
  info("Performing symmetric interpolation test for Cylindrical scanner");
  auto time_frame_def = TimeFrameDefinitions();
  time_frame_def.set_num_time_frames(1);
  time_frame_def.set_time_frame(1, 0, 1e9);
  auto exam_info = ExamInfo();
  exam_info.set_high_energy_thres(650);
  exam_info.set_low_energy_thres(425);
  exam_info.set_time_frame_definitions(time_frame_def);

  // define the original scanner and a downsampled one, as it would be used for scatter simulation
  auto scanner = Scanner(Scanner::User_defined_scanner,
                         "Some_symmetric_scanner",
                         192,
                         30,
                         150,
                         150,
                         127,
                         4.3,
                         4.0,
                         2.0,
                         -0.38956 /* 0.0 */,
                         5,
                         4,
                         6,
                         6,
                         1,
                         1,
                         1,
                         0.17,
                         511,
                         -1,
                         01.F,
                         -1.F,
                         "Cylindrical",
                         4.0,
                         4.0,
                         24.0,
                         24.0);
  auto downsampled_scanner = Scanner(Scanner::User_defined_scanner,
                                     "Some_symmetric_scanner",
                                     64,
                                     6,
                                     int(150 * 64 / 192),
                                     int(150 * 64 / 192),
                                     127,
                                     4.3,
                                     20.0,
                                     133 * 3.14 / 64,
                                     -0.38956 /* 0.0 */,
                                     1,
                                     1,
                                     6,
                                     64,
                                     1,
                                     1,
                                     1,
                                     0.17,
                                     511,
                                     -1,
                                     01.F,
                                     -1.F,
                                     "Cylindrical",
                                     20.0,
                                     12.0,
                                     120.0,
                                     72.0);

  auto proj_data_info = shared_ptr<ProjDataInfo>(
      std::move(ProjDataInfo::construct_proj_data_info(std::make_shared<Scanner>(scanner), 1, 29, 96, 150, false)));
  auto downsampled_proj_data_info = shared_ptr<ProjDataInfo>(std::move(ProjDataInfo::construct_proj_data_info(
      std::make_shared<Scanner>(downsampled_scanner), 1, 0, 32, int(150 * 64 / 192), false)));

  auto proj_data = ProjDataInMemory(std::make_shared<ExamInfo>(exam_info), proj_data_info);
  auto downsampled_proj_data = ProjDataInMemory(std::make_shared<ExamInfo>(exam_info), downsampled_proj_data_info);

  // define a cylinder precisely in the middle of the FOV, such that symmetry can be used for validation
  auto emission_map = VoxelsOnCartesianGrid<float>(*downsampled_proj_data_info, 1);
  make_symmetric_object(emission_map);
  write_to_file("downsampled_cylinder_map_cyl", emission_map);

  // project the cylinder onto the downsampled scanner proj data
  auto pm = ProjMatrixByBinUsingRayTracing();
  pm.set_use_actual_detector_boundaries(true);
  pm.enable_cache(false);
  auto forw_proj = ForwardProjectorByBinUsingProjMatrixByBin(std::make_shared<ProjMatrixByBinUsingRayTracing>(pm));
  forw_proj.set_up(downsampled_proj_data_info, std::make_shared<VoxelsOnCartesianGrid<float>>(emission_map));
  auto downsampled_model_sino = ProjDataInMemory(downsampled_proj_data);
  downsampled_model_sino.fill(0);
  forw_proj.forward_project(downsampled_model_sino, emission_map);

  // interpolate the downsampled proj data to the original scanner size and fill in oblique sinograms
  auto interpolated_direct_proj_data = ProjDataInMemory(proj_data);
  interpolate_projdata(interpolated_direct_proj_data, downsampled_model_sino, BSpline::linear, false);
  auto interpolated_proj_data = ProjDataInMemory(proj_data);
  inverse_SSRB(interpolated_proj_data, interpolated_direct_proj_data);

  // write the proj data to file
  downsampled_model_sino.write_to_file("downsampled_sino_cyl.hs");
  interpolated_proj_data.write_to_file("interpolated_sino_cyl.hs");

  // use symmetry to check that there are no significant errors in the interpolation
  check_symmetry(interpolated_proj_data.get_segment_by_sinogram(0));
}

void
InterpolationTests::scatter_interpolation_test_blocks_asymmetric()
{
  info("Performing asymmetric interpolation test for BlocksOnCylindrical scanner");
  auto time_frame_def = TimeFrameDefinitions();
  time_frame_def.set_num_time_frames(1);
  time_frame_def.set_time_frame(1, 0, 1e9);
  auto exam_info = ExamInfo();
  exam_info.set_high_energy_thres(650);
  exam_info.set_low_energy_thres(425);
  exam_info.set_time_frame_definitions(time_frame_def);

  // define the original scanner and a downsampled one, as it would be used for scatter simulation
  auto scanner = Scanner(Scanner::User_defined_scanner,
                         "Some_symmetric_scanner",
                         96,
                         30,
                         150,
                         150,
                         127,
                         4.3,
                         4.0,
                         8.0,
                         -0.38956 /* 0.0 */,
                         5,
                         1,
                         6,
                         6,
                         1,
                         1,
                         1,
                         0.17,
                         511,
                         -1,
                         01.F,
                         -1.F,
                         "BlocksOnCylindrical",
                         4.0,
                         16.0,
                         24.0,
                         96.0);
  auto downsampled_scanner = Scanner(Scanner::User_defined_scanner,
                                     "Some_symmetric_scanner",
                                     96,
                                     12,
                                     150,
                                     150,
                                     127,
                                     4.3,
                                     10.0,
                                     8.0,
                                     -0.38956 /* 0.0 */,
                                     1,
                                     1,
                                     12,
                                     6,
                                     1,
                                     1,
                                     1,
                                     0.17,
                                     511,
                                     -1,
                                     01.F,
                                     -1.F,
                                     "BlocksOnCylindrical",
                                     10.0,
                                     16.0,
                                     120.0,
                                     96.0);

  auto proj_data_info = shared_ptr<ProjDataInfo>(
      std::move(ProjDataInfo::construct_proj_data_info(std::make_shared<Scanner>(scanner), 1, 29, 48, 75, false)));
  auto downsampled_proj_data_info = shared_ptr<ProjDataInfo>(
      std::move(ProjDataInfo::construct_proj_data_info(std::make_shared<Scanner>(downsampled_scanner), 1, 0, 48, 75, false)));

  auto proj_data = ProjDataInMemory(std::make_shared<ExamInfo>(exam_info), proj_data_info);
  auto downsampled_proj_data = ProjDataInMemory(std::make_shared<ExamInfo>(exam_info), downsampled_proj_data_info);

  // define a cylinder precisely in the middle of the FOV, such that symmetry can be used for validation
  auto emission_map = VoxelsOnCartesianGrid<float>(*proj_data_info, 1);
  auto cyl_map = VoxelsOnCartesianGrid<float>(*proj_data_info, 1);
  auto cylinder = EllipsoidalCylinder(40, 40, 20, CartesianCoordinate3D<float>(90, 100, 0));
  cylinder.construct_volume(cyl_map, CartesianCoordinate3D<int>(1, 1, 1));
  auto box = Box3D(20, 20, 20, CartesianCoordinate3D<float>(40, -20, 70));
  box.construct_volume(emission_map, CartesianCoordinate3D<int>(1, 1, 1));
  emission_map += cyl_map;

  // project the cylinder onto the full-scale scanner proj data
  auto pm = ProjMatrixByBinUsingRayTracing();
  pm.set_use_actual_detector_boundaries(true);
  pm.enable_cache(false);
  auto forw_proj = ForwardProjectorByBinUsingProjMatrixByBin(std::make_shared<ProjMatrixByBinUsingRayTracing>(pm));
  forw_proj.set_up(proj_data_info, std::make_shared<VoxelsOnCartesianGrid<float>>(emission_map));
  auto full_size_model_sino = ProjDataInMemory(proj_data);
  full_size_model_sino.fill(0);
  forw_proj.forward_project(full_size_model_sino, emission_map);

  // also project onto the downsampled scanner
  emission_map = VoxelsOnCartesianGrid<float>(*downsampled_proj_data_info, 1);
  cyl_map = VoxelsOnCartesianGrid<float>(*downsampled_proj_data_info, 1);
  cylinder.construct_volume(cyl_map, CartesianCoordinate3D<int>(1, 1, 1));
  box.construct_volume(emission_map, CartesianCoordinate3D<int>(1, 1, 1));
  emission_map += cyl_map;
  forw_proj.set_up(downsampled_proj_data_info, std::make_shared<VoxelsOnCartesianGrid<float>>(emission_map));
  auto downsampled_model_sino = ProjDataInMemory(downsampled_proj_data);
  downsampled_model_sino.fill(0);
  forw_proj.forward_project(downsampled_model_sino, emission_map);

  // interpolate the downsampled proj data to the original scanner size and fill in oblique sinograms
  auto interpolated_direct_proj_data = ProjDataInMemory(proj_data);
  interpolate_projdata(interpolated_direct_proj_data, downsampled_model_sino, BSpline::linear, false);
  auto interpolated_proj_data = ProjDataInMemory(proj_data);
  inverse_SSRB(interpolated_proj_data, interpolated_direct_proj_data);

  // write the proj data to file
  downsampled_model_sino.write_to_file("downsampled_sino_asym.hs");
  full_size_model_sino.write_to_file("full_size_sino_asym.hs");
  interpolated_proj_data.write_to_file("interpolated_sino_asym.hs");

  // compare to ground truth
  compare_segment_shape(full_size_model_sino.get_segment_by_sinogram(0), interpolated_proj_data.get_segment_by_sinogram(0), 2);
}

void
InterpolationTests::scatter_interpolation_test_cyl_asymmetric()
{
  info("Performing asymmetric interpolation test for Cylindrical scanner");
  auto time_frame_def = TimeFrameDefinitions();
  time_frame_def.set_num_time_frames(1);
  time_frame_def.set_time_frame(1, 0, 1e9);
  auto exam_info = ExamInfo();
  exam_info.set_high_energy_thres(650);
  exam_info.set_low_energy_thres(425);
  exam_info.set_time_frame_definitions(time_frame_def);

  // define the original scanner and a downsampled one, as it would be used for scatter simulation
  auto scanner = Scanner(Scanner::User_defined_scanner,
                         "Some_symmetric_scanner",
                         96,
                         30,
                         150,
                         150,
                         127,
                         4.3,
                         4.0,
                         8.0,
                         -0.38956 /* 0.0 */,
                         5,
                         1,
                         6,
                         6,
                         1,
                         1,
                         1,
                         0.17,
                         511,
                         -1,
                         01.F,
                         -1.F,
                         "Cylindrical",
                         4.0,
                         16.0,
                         24.0,
                         96.0);
  auto downsampled_scanner = Scanner(Scanner::User_defined_scanner,
                                     "Some_symmetric_scanner",
                                     64,
                                     12,
                                     150,
                                     150,
                                     127,
                                     4.3,
                                     10.0,
                                     133 * 3.14 / 64,
                                     -0.38956 /* 0.0 */,
                                     1,
                                     1,
                                     12,
                                     64,
                                     1,
                                     1,
                                     1,
                                     0.17,
                                     511,
                                     -1,
                                     01.F,
                                     -1.F,
                                     "Cylindrical",
                                     10.0,
                                     12.0,
                                     60.0,
                                     72.0);

  auto proj_data_info = shared_ptr<ProjDataInfo>(std::move(
      ProjDataInfo::construct_proj_data_info(std::make_shared<Scanner>(scanner), 1, 29, 48, int(150 * 96 / 192), false)));
  auto downsampled_proj_data_info = shared_ptr<ProjDataInfo>(std::move(ProjDataInfo::construct_proj_data_info(
      std::make_shared<Scanner>(downsampled_scanner), 1, 0, 32, int(150 * 64 / 192), false)));

  auto proj_data = ProjDataInMemory(std::make_shared<ExamInfo>(exam_info), proj_data_info);
  auto downsampled_proj_data = ProjDataInMemory(std::make_shared<ExamInfo>(exam_info), downsampled_proj_data_info);

  // define asymetric object
  auto emission_map = VoxelsOnCartesianGrid<float>(*proj_data_info, 1);
  auto cyl_map = VoxelsOnCartesianGrid<float>(*proj_data_info, 1);
  auto cylinder = EllipsoidalCylinder(40, 40, 20, CartesianCoordinate3D<float>(90, 100, 0));
  cylinder.construct_volume(cyl_map, CartesianCoordinate3D<int>(1, 1, 1));
  auto box = Box3D(20, 20, 20, CartesianCoordinate3D<float>(40, -20, 70));
  box.construct_volume(emission_map, CartesianCoordinate3D<int>(1, 1, 1));
  emission_map += cyl_map;

  // project the cylinder onto the full-scale scanner proj data
  auto pm = ProjMatrixByBinUsingRayTracing();
  pm.set_use_actual_detector_boundaries(true);
  pm.enable_cache(false);
  auto forw_proj = ForwardProjectorByBinUsingProjMatrixByBin(std::make_shared<ProjMatrixByBinUsingRayTracing>(pm));
  forw_proj.set_up(proj_data_info, std::make_shared<VoxelsOnCartesianGrid<float>>(emission_map));
  auto full_size_model_sino = ProjDataInMemory(proj_data);
  full_size_model_sino.fill(0);
  forw_proj.forward_project(full_size_model_sino, emission_map);

  // also project onto the downsampled scanner
  emission_map = VoxelsOnCartesianGrid<float>(*downsampled_proj_data_info, 1);
  cyl_map = VoxelsOnCartesianGrid<float>(*downsampled_proj_data_info, 1);
  cylinder.construct_volume(cyl_map, CartesianCoordinate3D<int>(1, 1, 1));
  box.construct_volume(emission_map, CartesianCoordinate3D<int>(1, 1, 1));
  emission_map += cyl_map;
  forw_proj.set_up(downsampled_proj_data_info, std::make_shared<VoxelsOnCartesianGrid<float>>(emission_map));
  auto downsampled_model_sino = ProjDataInMemory(downsampled_proj_data);
  downsampled_model_sino.fill(0);
  forw_proj.forward_project(downsampled_model_sino, emission_map);

  // interpolate the downsampled proj data to the original scanner size and fill in oblique sinograms
  auto interpolated_direct_proj_data = ProjDataInMemory(proj_data);
  interpolate_projdata(interpolated_direct_proj_data, downsampled_model_sino, BSpline::linear, false);
  auto interpolated_proj_data = ProjDataInMemory(proj_data);
  inverse_SSRB(interpolated_proj_data, interpolated_direct_proj_data);

  // write the proj data to file
  downsampled_model_sino.write_to_file("downsampled_sino_cyl_asym.hs");
  full_size_model_sino.write_to_file("full_size_sino_cyl_asym.hs");
  interpolated_proj_data.write_to_file("interpolated_sino_cyl_asym.hs");

  // compare to ground truth
  compare_segment_shape(full_size_model_sino.get_segment_by_sinogram(0), interpolated_proj_data.get_segment_by_sinogram(0), 2);
}

void
InterpolationTests::scatter_interpolation_test_blocks_downsampled()
{
  info("Performing downampled interpolation test for BlocksOnCylindrical scanner");
  auto time_frame_def = TimeFrameDefinitions();
  time_frame_def.set_num_time_frames(1);
  time_frame_def.set_time_frame(1, 0, 1e9);
  auto exam_info = ExamInfo();
  exam_info.set_high_energy_thres(650);
  exam_info.set_low_energy_thres(425);
  exam_info.set_time_frame_definitions(time_frame_def);

  // define the original scanner and a downsampled one, as it would be used for scatter simulation
  auto scanner = Scanner(Scanner::User_defined_scanner,
                         "NeuroLF_10mm",
                         256,
                         48,
                         180,
                         180,
                         134,
                         6.99,
                         3.313,
                         1.6565,
                         -3.1091819,
                         6,
                         4,
                         8,
                         8,
                         1,
                         1,
                         1,
                         0.14,
                         511,
                         1,
                         0,
                         500,
                         "BlocksOnCylindrical",
                         3.313,
                         3.313,
                         27.36,
                         27.36);
  auto downsampled_scanner = Scanner(Scanner::User_defined_scanner,
                                     "NeuroLF_10mm",
                                     128,
                                     8,
                                     91,
                                     180,
                                     127,
                                     6.99,
                                     22.8559,
                                     3.46,
                                     -3.1091819,
                                     1,
                                     1,
                                     8,
                                     16,
                                     1,
                                     1,
                                     1,
                                     0.17,
                                     511,
                                     1,
                                     0,
                                     500,
                                     "BlocksOnCylindrical",
                                     22.8559,
                                     7.01807,
                                     182.848,
                                     112.290);

  auto proj_data_info = shared_ptr<ProjDataInfo>(
      std::move(ProjDataInfo::construct_proj_data_info(std::make_shared<Scanner>(scanner), 1, 47, 128, 180, false)));
  auto downsampled_proj_data_info = shared_ptr<ProjDataInfo>(
      std::move(ProjDataInfo::construct_proj_data_info(std::make_shared<Scanner>(downsampled_scanner), 1, 0, 64, 91, false)));

  auto proj_data = ProjDataInMemory(std::make_shared<ExamInfo>(exam_info), proj_data_info);
  // auto downsampled_proj_data = ProjDataInMemory(std::make_shared<ExamInfo>(exam_info), downsampled_proj_data_info);

  // // define a cylinder precisely in the middle of the FOV, such that symmetry can be used for validation
  // auto emission_map = VoxelsOnCartesianGrid<float>(*proj_data_info, 1);
  // auto cyl_map = VoxelsOnCartesianGrid<float>(*proj_data_info, 1);
  // auto cylinder = EllipsoidalCylinder(40, 40, 20, CartesianCoordinate3D<float>(90, 100, 0));
  // cylinder.construct_volume(cyl_map, CartesianCoordinate3D<int>(1, 1, 1));
  // auto box = Box3D(20, 20, 20, CartesianCoordinate3D<float>(40, -20, 70));
  // box.construct_volume(emission_map, CartesianCoordinate3D<int>(1, 1, 1));
  // emission_map += cyl_map;

  // // project the cylinder onto the full-scale scanner proj data
  // auto pm = ProjMatrixByBinUsingRayTracing();
  // pm.set_use_actual_detector_boundaries(true);
  // pm.enable_cache(false);
  // auto forw_proj = ForwardProjectorByBinUsingProjMatrixByBin(std::make_shared<ProjMatrixByBinUsingRayTracing>(pm));
  // forw_proj.set_up(proj_data_info, std::make_shared<VoxelsOnCartesianGrid<float>>(emission_map));
  // auto full_size_model_sino = ProjDataInMemory(proj_data);
  // full_size_model_sino.fill(0);
  // forw_proj.forward_project(full_size_model_sino, emission_map);

  // // also project onto the downsampled scanner
  // emission_map = VoxelsOnCartesianGrid<float>(*downsampled_proj_data_info, 1);
  // cyl_map = VoxelsOnCartesianGrid<float>(*downsampled_proj_data_info, 1);
  // cylinder.construct_volume(cyl_map, CartesianCoordinate3D<int>(1, 1, 1));
  // box.construct_volume(emission_map, CartesianCoordinate3D<int>(1, 1, 1));
  // emission_map += cyl_map;
  // forw_proj.set_up(downsampled_proj_data_info, std::make_shared<VoxelsOnCartesianGrid<float>>(emission_map));
  // auto downsampled_model_sino = ProjDataInMemory(downsampled_proj_data);
  // downsampled_model_sino.fill(0);
  // forw_proj.forward_project(downsampled_model_sino, emission_map);

  // // write the proj data to file
  // downsampled_model_sino.write_to_file("downsampled_sino_asym.hs");
  // full_size_model_sino.write_to_file("full_size_sino_asym.hs");

  auto downsampled_model_sino = std::make_shared<ProjDataInMemory>(
      *(ProjDataInMemory::read_from_file("/workspace/stir-public/test_data/downsampled_sino_neurolf.hs")));

  // interpolate the downsampled proj data to the original scanner size and fill in oblique sinograms
  auto interpolated_direct_proj_data = ProjDataInMemory(proj_data);
  interpolate_projdata(interpolated_direct_proj_data, *downsampled_model_sino, BSpline::linear, false);
  auto interpolated_proj_data = ProjDataInMemory(proj_data);
  inverse_SSRB(interpolated_proj_data, interpolated_direct_proj_data);

  // write the proj data to file
  interpolated_proj_data.write_to_file("interpolated_sino_asym.hs");

  // compare to ground truth
  // compare_segment_shape(full_size_model_sino.get_segment_by_sinogram(0), interpolated_proj_data.get_segment_by_sinogram(0), 2);
}

void
InterpolationTests::run_tests()
{
  scatter_interpolation_test_blocks();
  scatter_interpolation_test_cyl();
  scatter_interpolation_test_blocks_asymmetric();
  scatter_interpolation_test_cyl_asymmetric();
  scatter_interpolation_test_blocks_downsampled();
}

END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int
main()
{
  Verbosity::set(1);
  InterpolationTests tests;
  tests.run_tests();
  return tests.main_return_value();
}
