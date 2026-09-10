/*
  Copyright 2023, Positrigo AG, Zurich
  Copyright 2024, 2026 University College London
  This file is part of STIR.

  SPDX-License-Identifier: Apache-2.0

  See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup test

  \brief Tests for stir::ProjData interpolation as used by the scatter estimation.

  \author Markus Jehl
  \author Kris Thielemans
*/

#ifndef NDEBUG
// set to high level of debugging
#  ifdef _DEBUG
#    undef _DEBUG
#  endif
#  define _DEBUG 2
#endif

#include "stir/ProjDataInfo.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
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
#include "stir/scatter/SingleScatterSimulation.h"
#include "stir/format.h"
#include "stir/extend_projdata.h"

#include <unordered_set>

#include "stir/RunTests.h"

START_NAMESPACE_STIR

class InterpolationTests : public RunTests
{
public:
  void run_tests() override;

private:
  shared_ptr<ExamInfo> exam_info_sptr;
  // need to call this first!
  void create_exam_info();
  void swap_segments_test();
  void extend_projdata_test();
  void scatter_interpolation_test_blocks();
  void scatter_interpolation_test_cyl(const bool do_3d = false);
  void scatter_interpolation_test_blocks_asymmetric();
  void scatter_interpolation_test_cyl_asymmetric(const bool do_3d = false);
  void scatter_interpolation_test_blocks_downsampled();
  void transaxial_upsampling_interpolation_test_blocks();

  void check_symmetry(const SegmentBySinogram<float>& segment,
                      const std::shared_ptr<SegmentBySinogram<float>> other_sptr = nullptr);
  void compare_segment(const SegmentBySinogram<float>& segment1, const SegmentBySinogram<float>& segment2, float maxDiff);
  void compare_segment_shape(const SegmentBySinogram<float>& shape_segment,
                             const SegmentBySinogram<float>& test_segment,
                             int dilation);
  //! forward project emission_map
  shared_ptr<ProjDataInMemory> create_data(const shared_ptr<const ProjDataInfo>& proj_data_info_sptr,
                                           const VoxelsOnCartesianGrid<float>& emission_map,
                                           const std::string& name);
  //! forward project emission_map to downsampled data and upsample
  shared_ptr<ProjDataInMemory> create_upsampled_data(const shared_ptr<const ProjDataInfo>& proj_data_info_sptr,
                                                     const shared_ptr<const ProjDataInfo> downsampled_proj_data_info_sptr,
                                                     const VoxelsOnCartesianGrid<float>& emission_map,
                                                     const std::string& suffix);
};

void
InterpolationTests::check_symmetry(const SegmentBySinogram<float>& segment,
                                   const std::shared_ptr<SegmentBySinogram<float>> other_sptr)
{
  bool with_opposite = (other_sptr != nullptr);

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
              auto voxel2 = with_opposite ? std::abs((*other_sptr)[decreasing_index][view][tang])
                                          : std::abs(segment[decreasing_index][view][tang]);
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

  if (!with_opposite)
    {
      // compare the first half of the views with the second half - even for the BlocksOnCylindrical scanner they should be
      // identical
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
}

void
InterpolationTests::swap_segments_test()
{
  info("Performing tests on swapping ProjData segment");
  auto small_scanner = Scanner(Scanner::User_defined_scanner,
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

  auto small_proj_data_info = shared_ptr<ProjDataInfo>(std::move(
      ProjDataInfo::construct_proj_data_info(std::make_shared<Scanner>(small_scanner), 1, 5, 32, int(150 * 64 / 192), false)));

  auto proj_data = ProjDataInMemory(this->exam_info_sptr, small_proj_data_info);

  int counter = 1;
  for (int i_seg = small_proj_data_info->get_min_segment_num(); i_seg <= small_proj_data_info->get_max_segment_num(); ++i_seg)
    {
      auto seg = proj_data.get_empty_segment_by_sinogram(i_seg);
      for (int i_axial = seg.get_min_axial_pos_num(); i_axial <= seg.get_max_axial_pos_num(); ++i_axial)
        {
          for (int i_view = seg.get_min_view_num(); i_view <= seg.get_max_view_num(); ++i_view)
            {
              for (int i_tang = seg.get_min_tangential_pos_num(); i_tang <= seg.get_max_tangential_pos_num(); ++i_tang)
                {
                  seg[i_axial][i_view][i_tang] = counter;
                  counter++;
                }
            }
        }
      proj_data.set_segment(seg);
    }

  auto no_arc = dynamic_pointer_cast<ProjDataInfoCylindricalNoArcCorr>(small_proj_data_info);

  for (int i_seg = small_proj_data_info->get_min_segment_num(); i_seg < 0; ++i_seg)
    {
      auto seg = proj_data.get_segment_by_sinogram(-i_seg);
      auto swapped = make_swapped_segment(seg, *no_arc, i_seg);
      std::unordered_set<float> seen_values;
      int num_duplicates = 0;
      int num_zero_fallback = 0;

      for (auto it = swapped.begin_all(); it != swapped.end_all(); ++it)
        {
          if (*it == 0.0F)
            {
              ++num_zero_fallback; // the deliberate out-of-range fallback in find_swapped_bin, not a real swapped value
              continue;
            }
          if (!seen_values.insert(*it).second)
            {
              ++num_duplicates;
              std::cerr << "value " << *it << " appears more than once in the swapped segment (segment " << i_seg << ")\n";
            }
        }

      check_if_equal(num_duplicates, 0, "The detector swap reused the same source bin for more than one destination bin.");
      check_if_equal(num_zero_fallback, 0, "During segment swapping bins were unclaimed.");
      check_if_equal(seg, swapped, "Segments are equal");
    }
}

void
InterpolationTests::extend_projdata_test()
{
  info("Performing tests on extending ProjData");
  auto small_scanner = Scanner(Scanner::User_defined_scanner,
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

  auto small_proj_data_info = shared_ptr<ProjDataInfo>(std::move(
      ProjDataInfo::construct_proj_data_info(std::make_shared<Scanner>(small_scanner), 1, 5, 32, int(150 * 64 / 192), false)));

  auto proj_data = ProjDataInMemory(this->exam_info_sptr, small_proj_data_info);

  int counter = 0;
  for (int i_seg = small_proj_data_info->get_min_segment_num(); i_seg <= small_proj_data_info->get_max_segment_num(); ++i_seg)
    {
      auto seg = proj_data.get_empty_segment_by_sinogram(i_seg);
      for (int i_axial = seg.get_min_axial_pos_num(); i_axial <= seg.get_max_axial_pos_num(); ++i_axial)
        {
          for (int i_view = seg.get_min_view_num(); i_view <= seg.get_max_view_num(); ++i_view)
            {
              for (int i_tang = seg.get_min_tangential_pos_num(); i_tang <= seg.get_max_tangential_pos_num(); ++i_tang)
                {
                  seg[i_axial][i_view][i_tang] = counter;
                  counter++;
                }
            }
        }
      proj_data.set_segment(seg);
    }

  auto no_arc = dynamic_pointer_cast<ProjDataInfoCylindricalNoArcCorr>(small_proj_data_info);

  for (int i_seg = small_proj_data_info->get_min_segment_num(); i_seg <= small_proj_data_info->get_min_segment_num(); ++i_seg)
    {
      auto seg = proj_data.get_segment_by_sinogram(i_seg);
      auto opp_seg = proj_data.get_segment_by_sinogram(-i_seg);
      auto swapped = make_swapped_segment(opp_seg, *no_arc, i_seg);

      int extend_views = seg.get_num_views() / 2;
      auto ext = extend_segment(seg, extend_views, 0, 0, &swapped);

      {
        int i_view_o = 0;
        const int max_view = swapped[0].get_max_index();
        for (int i_view = ext[0].get_min_index(); i_view < 0; ++i_view, i_view_o++)
          {
            // Don't compare the first tang pos, it cannot align after full rotation
            for (int i_tang = ext[0][0].get_min_index() + 1; i_tang <= ext[0][0].get_max_index(); ++i_tang)
              {
                const int src_view = max_view - extend_views + i_view_o + 1;
                float diff = ext[0][i_view][i_tang] - swapped[0][src_view][-i_tang];
                check_if_equal(diff, 0.0, "The north extended sinogram does not match the swapped");
              }
          }
      }

      {
        for (int i_view = 0; i_view <= seg.get_max_view_num(); ++i_view)
          {
            for (int i_tang = ext[0][0].get_min_index() + 1; i_tang <= ext[0][0].get_max_index(); ++i_tang)
              {
                // std::cout << i_view << " " << i_tang << " " << " : ";
                // std::cout << ext[0][i_view][i_tang]<< " "
                //           << seg[0][i_view][i_tang] << std::endl;
                float diff = ext[0][i_view][i_tang] - seg[0][i_view][i_tang];
                check_if_equal(diff, 0.0, "The central extended sinogram match the original");
              }
          }
      }

      {
        int i_view_o = 0;
        for (int i_view = seg.get_num_views(); i_view < ext[0].get_max_index(); ++i_view, i_view_o++)
          {
            for (int i_tang = ext[0][0].get_min_index() + 1; i_tang <= ext[0][0].get_max_index(); ++i_tang)
              {
                // std::cout << i_view << " " << i_tang << " " << i_view_o << " : ";
                // std::cout << ext[0][i_view][i_tang] << " " << swapped[0][i_view_o][-i_tang] << std::endl;
                float diff = ext[0][i_view][i_tang] - swapped[0][i_view_o][-i_tang];
                check_if_equal(diff, 0.0, "The south extended sinogram matches the swapped");
              }
          }
      }
    }
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
                                          int dilation)
{
  const float test_threshold = 0.1 * test_segment.find_max();
  // const float shape_threshold = 0.1 * shape_segment.find_max();

  // compute difference and compare against empirically found value from visually validated sinograms
  auto sumVoxelsOutsideMask = 0U;
  for (auto axial = test_segment.get_min_axial_pos_num(); axial <= test_segment.get_max_axial_pos_num(); axial++)
    {
      for (auto view = test_segment.get_min_view_num(); view <= test_segment.get_max_view_num(); view++)
        {
          for (auto tang = test_segment.get_min_tangential_pos_num(); tang <= test_segment.get_max_tangential_pos_num(); tang++)
            {
              if (test_segment[axial][view][tang] < test_threshold)
                continue;

              // now go through the dilation neighbourhood of the voxel to see if it is near a non-zero voxel
              bool isNearNonZero = false;
              for (auto axialShape = std::max(axial - dilation, test_segment.get_min_axial_pos_num());
                   axialShape <= std::min(axial + dilation, test_segment.get_max_axial_pos_num());
                   axialShape++)
                {
                  for (auto viewShape = std::max(view - dilation, test_segment.get_min_view_num());
                       viewShape <= std::min(view + dilation, test_segment.get_max_view_num());
                       viewShape++)
                    {
                      for (auto tangShape = std::max(tang - dilation, test_segment.get_min_tangential_pos_num());
                           tangShape <= std::min(tang + dilation, test_segment.get_max_tangential_pos_num());
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
InterpolationTests::create_exam_info()
{
  auto time_frame_def = TimeFrameDefinitions();
  time_frame_def.set_num_time_frames(1);
  time_frame_def.set_time_frame(1, 0, 1e9);
  this->exam_info_sptr = std::make_shared<ExamInfo>(ImagingModality::PT);
  this->exam_info_sptr->set_high_energy_thres(650);
  this->exam_info_sptr->set_low_energy_thres(425);
  this->exam_info_sptr->set_time_frame_definitions(time_frame_def);
}

shared_ptr<ProjDataInMemory>
InterpolationTests::create_data(const shared_ptr<const ProjDataInfo>& proj_data_info_sptr,
                                const VoxelsOnCartesianGrid<float>& emission_map,
                                const std::string& name)
{
  // project the cylinder onto the full-scale scanner proj data
  auto pm = ProjMatrixByBinUsingRayTracing();
  pm.set_use_actual_detector_boundaries(true);
  pm.enable_cache(false);
  auto forw_proj = ForwardProjectorByBinUsingProjMatrixByBin(std::make_shared<ProjMatrixByBinUsingRayTracing>(pm));
  forw_proj.set_up(proj_data_info_sptr, std::make_shared<VoxelsOnCartesianGrid<float>>(emission_map));
  auto full_size_model_sino_sptr = std::make_shared<ProjDataInMemory>(this->exam_info_sptr, proj_data_info_sptr);
  forw_proj.forward_project(*full_size_model_sino_sptr, emission_map);

  if (!name.empty())
    full_size_model_sino_sptr->write_to_file(name + ".hs");
  return full_size_model_sino_sptr;
}

shared_ptr<ProjDataInMemory>
InterpolationTests::create_upsampled_data(const shared_ptr<const ProjDataInfo>& proj_data_info_sptr,
                                          const shared_ptr<const ProjDataInfo> downsampled_proj_data_info_sptr,
                                          const VoxelsOnCartesianGrid<float>& emission_map,
                                          const std::string& suffix)
{
  auto downsampled_proj_data = this->create_data(downsampled_proj_data_info_sptr, emission_map, "downsampled_sino" + suffix);

  // interpolate the downsampled proj data to the original scanner size and fill in oblique sinograms
  // TODO reduce_segment
  if (downsampled_proj_data->get_num_segments() == 1)
    {
      auto interpolated_direct_proj_data = ProjDataInMemory(this->exam_info_sptr, proj_data_info_sptr);
      interpolate_projdata(interpolated_direct_proj_data, *downsampled_proj_data, BSpline::linear, false);
      auto interpolated_proj_data_sptr = std::make_shared<ProjDataInMemory>(this->exam_info_sptr, proj_data_info_sptr);
      inverse_SSRB(*interpolated_proj_data_sptr, interpolated_direct_proj_data);

      // write the proj data to file
      interpolated_proj_data_sptr->write_to_file("interpolated_sino" + suffix + ".hs");

      return interpolated_proj_data_sptr;
    }
  else
    {
      auto interpolated_proj_data_sptr = std::make_shared<ProjDataInMemory>(this->exam_info_sptr, proj_data_info_sptr);
      interpolate_projdata_3d(*interpolated_proj_data_sptr, *downsampled_proj_data, BSpline::linear, false);
      // write the proj data to file
      interpolated_proj_data_sptr->write_to_file("interpolated_sino" + suffix + ".hs");

      return interpolated_proj_data_sptr;
    }
}

void
InterpolationTests::scatter_interpolation_test_blocks()
{
  info("Performing symmetric interpolation test for BlocksOnCylindrical scanner");

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

  // define a cylinder precisely in the middle of the FOV, such that symmetry can be used for validation
  auto emission_map = VoxelsOnCartesianGrid<float>(this->exam_info_sptr, *downsampled_proj_data_info, 1);
  make_symmetric_object(emission_map);
  write_to_file("downsampled_cylinder_map", emission_map);

  auto interpolated_proj_data_sptr
      = this->create_upsampled_data(proj_data_info, downsampled_proj_data_info, emission_map, "_block");

  // use symmetry to check that there are no significant errors in the interpolation
  check_symmetry(interpolated_proj_data_sptr->get_segment_by_sinogram(0));
}

void
InterpolationTests::scatter_interpolation_test_cyl(const bool do_3d)
{
  info("Performing symmetric interpolation test for Cylindrical scanner");

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
  int downsampled_rings = do_3d ? 5 : 0;
  auto downsampled_proj_data_info = shared_ptr<ProjDataInfo>(std::move(ProjDataInfo::construct_proj_data_info(
      std::make_shared<Scanner>(downsampled_scanner), 1, downsampled_rings, 32, int(150 * 64 / 192), false)));

  auto proj_data = ProjDataInMemory(this->exam_info_sptr, proj_data_info);

  // define a cylinder precisely in the middle of the FOV, such that symmetry can be used for validation
  auto emission_map = VoxelsOnCartesianGrid<float>(this->exam_info_sptr, *downsampled_proj_data_info, 1);
  make_symmetric_object(emission_map);
  write_to_file("downsampled_cylinder_map_cyl", emission_map);

  auto interpolated_proj_data_sptr
      = this->create_upsampled_data(proj_data_info, downsampled_proj_data_info, emission_map, "_cyl");

  const auto proj_data_info_no_arc_corr_sptr
      = dynamic_pointer_cast<const ProjDataInfoCylindricalNoArcCorr>(interpolated_proj_data_sptr->get_proj_data_info_sptr());
  if (!proj_data_info_no_arc_corr_sptr)
    error("Expected the in projection data info to be a ProjDataInfoCylindricalNoArcCorr.");

  for (int i_seg = interpolated_proj_data_sptr->get_min_segment_num(); i_seg < 0; ++i_seg)
    {
      auto opp_sptr = std::make_shared<SegmentBySinogram<float>>(interpolated_proj_data_sptr->get_segment_by_sinogram(-i_seg));
      // const SegmentBySinogram<float> swapped_opposite
      //     = make_swapped_segment(opposite_segment, *proj_data_info_no_arc_corr_sptr, i_seg);

      // use symmetry to check that there are no significant errors in the interpolation
      check_symmetry(interpolated_proj_data_sptr->get_segment_by_sinogram(i_seg), opp_sptr);
    }
}

void
InterpolationTests::scatter_interpolation_test_blocks_asymmetric()
{
  info("Performing asymmetric interpolation test for BlocksOnCylindrical scanner");

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

  auto proj_data = ProjDataInMemory(this->exam_info_sptr, proj_data_info);
  auto downsampled_proj_data = ProjDataInMemory(exam_info_sptr, downsampled_proj_data_info);

  auto emission_map = VoxelsOnCartesianGrid<float>(this->exam_info_sptr, *proj_data_info, 1);
  auto cyl_map = VoxelsOnCartesianGrid<float>(this->exam_info_sptr, *proj_data_info, 1);
  auto cylinder = EllipsoidalCylinder(40, 40, 20, CartesianCoordinate3D<float>(90, 100, 0));
  cylinder.construct_volume(cyl_map, CartesianCoordinate3D<int>(1, 1, 1));
  auto box = Box3D(20, 20, 20, CartesianCoordinate3D<float>(40, -20, 70));
  box.construct_volume(emission_map, CartesianCoordinate3D<int>(1, 1, 1));
  emission_map += cyl_map;

  // project the cylinder onto the full-scale scanner proj data
  auto full_size_model_sino_sptr = this->create_data(proj_data_info, emission_map, "full_size_sino_asym_block");

  // also project down-sampled, and upsample
  emission_map = VoxelsOnCartesianGrid<float>(this->exam_info_sptr, *downsampled_proj_data_info, 1);
  cyl_map = VoxelsOnCartesianGrid<float>(this->exam_info_sptr, *downsampled_proj_data_info, 1);
  cylinder.construct_volume(cyl_map, CartesianCoordinate3D<int>(1, 1, 1));
  box.construct_volume(emission_map, CartesianCoordinate3D<int>(1, 1, 1));
  emission_map += cyl_map;
  auto interpolated_proj_data_sptr
      = this->create_upsampled_data(proj_data_info, downsampled_proj_data_info, emission_map, "asym_block");

  // compare to ground truth
  compare_segment_shape(
      full_size_model_sino_sptr->get_segment_by_sinogram(0), interpolated_proj_data_sptr->get_segment_by_sinogram(0), 2);
}

void
InterpolationTests::scatter_interpolation_test_cyl_asymmetric(const bool do_3d)
{
  info("Performing asymmetric interpolation test for Cylindrical scanner");

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

  int downsampled_rings = do_3d ? 11 : 0;
  auto proj_data_info = shared_ptr<ProjDataInfo>(std::move(
      ProjDataInfo::construct_proj_data_info(std::make_shared<Scanner>(scanner), 1, 29, 48, int(150 * 96 / 192), false)));
  auto downsampled_proj_data_info = shared_ptr<ProjDataInfo>(std::move(ProjDataInfo::construct_proj_data_info(
      std::make_shared<Scanner>(downsampled_scanner), 1, downsampled_rings, 32, int(150 * 64 / 192), false)));

  // define asymetric object
  auto emission_map = VoxelsOnCartesianGrid<float>(this->exam_info_sptr, *proj_data_info, 1);
  auto cyl_map = VoxelsOnCartesianGrid<float>(this->exam_info_sptr, *proj_data_info, 1);
  auto cylinder = EllipsoidalCylinder(40, 40, 20, CartesianCoordinate3D<float>(90, 100, 0));
  cylinder.construct_volume(cyl_map, CartesianCoordinate3D<int>(1, 1, 1));
  auto box = Box3D(20, 20, 20, CartesianCoordinate3D<float>(40, -20, 70));
  box.construct_volume(emission_map, CartesianCoordinate3D<int>(1, 1, 1));
  emission_map += cyl_map;

  // project the cylinder onto the full-scale scanner proj data
  auto full_size_model_sino_sptr = this->create_data(proj_data_info, emission_map, "full_size_sino_asym_cyl");

  // also project down-sampled, and upsample
  emission_map = VoxelsOnCartesianGrid<float>(this->exam_info_sptr, *downsampled_proj_data_info, 1);
  cyl_map = VoxelsOnCartesianGrid<float>(this->exam_info_sptr, *downsampled_proj_data_info, 1);
  cylinder.construct_volume(cyl_map, CartesianCoordinate3D<int>(1, 1, 1));
  box.construct_volume(emission_map, CartesianCoordinate3D<int>(1, 1, 1));
  emission_map += cyl_map;
  auto interpolated_proj_data_sptr
      = this->create_upsampled_data(proj_data_info, downsampled_proj_data_info, emission_map, "asym_cyl");

  if (downsampled_proj_data_info->get_num_segments() == 1)
    {
      // compare to ground truth
      compare_segment_shape(
          full_size_model_sino_sptr->get_segment_by_sinogram(0), interpolated_proj_data_sptr->get_segment_by_sinogram(0), 2);
    }
  else
    {
      for (int i_seg = proj_data_info->get_min_segment_num(); i_seg <= proj_data_info->get_max_segment_num(); ++i_seg)
        {
          // compare to ground truth
          std::cout << i_seg << std::endl;
          compare_segment_shape(full_size_model_sino_sptr->get_segment_by_sinogram(i_seg),
                                interpolated_proj_data_sptr->get_segment_by_sinogram(i_seg),
                                2);
        }
    }
}

void
InterpolationTests::scatter_interpolation_test_blocks_downsampled()
{
  info("Performing downampled interpolation test for BlocksOnCylindrical scanner");

  // define the original scanner and a downsampled one, as it would be used for scatter simulation
  auto scanner = Scanner(Scanner::User_defined_scanner,
                         "Some_BlocksOnCylindrical_Scanner",
                         96,
                         30,
                         int(150 * 96 / 192),
                         int(150 * 96 / 192),
                         127,
                         6.5,
                         3.313,
                         4.156,
                         -3.1091819,
                         5,
                         3,
                         6,
                         4,
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
                         7.0,
                         20.0,
                         29.0);
  auto downsampled_scanner = Scanner(Scanner::User_defined_scanner,
                                     "Some_Downsampled_BlocksOnCylindrical_Scanner",
                                     64,
                                     8,
                                     int(150 * 64 / 192 + 1),
                                     int(150 * 64 / 192 + 1),
                                     127,
                                     6.5,
                                     16.652,
                                     6.234,
                                     -3.1091819,
                                     1,
                                     1,
                                     8,
                                     8,
                                     1,
                                     1,
                                     1,
                                     0.17,
                                     511,
                                     1,
                                     0,
                                     500,
                                     "BlocksOnCylindrical",
                                     13.795,
                                     15.4286,
                                     112.0,
                                     125.0);

  auto proj_data_info = shared_ptr<ProjDataInfo>(std::move(
      ProjDataInfo::construct_proj_data_info(std::make_shared<Scanner>(scanner), 1, 29, 48, int(150 * 96 / 192), false)));
  auto downsampled_proj_data_info = shared_ptr<ProjDataInfo>(std::move(ProjDataInfo::construct_proj_data_info(
      std::make_shared<Scanner>(downsampled_scanner), 1, 0, 32, int(150 * 64 / 192 + 1), false)));

  auto proj_data = ProjDataInMemory(exam_info_sptr, proj_data_info);
  auto downsampled_proj_data = ProjDataInMemory(exam_info_sptr, downsampled_proj_data_info);

  // define a cylinder and a box that are off-centre, such that the shapes in the sinogram can be compared
  auto emission_map = VoxelsOnCartesianGrid<float>(this->exam_info_sptr, *proj_data_info, 1);
  auto cyl_map = VoxelsOnCartesianGrid<float>(this->exam_info_sptr, *proj_data_info, 1);
  auto cylinder = EllipsoidalCylinder(40, 40, 20, CartesianCoordinate3D<float>(80, 100, 0));
  cylinder.construct_volume(cyl_map, CartesianCoordinate3D<int>(1, 1, 1));
  auto box = Box3D(20, 20, 20, CartesianCoordinate3D<float>(30, -20, 70));
  box.construct_volume(emission_map, CartesianCoordinate3D<int>(1, 1, 1));
  emission_map += cyl_map;

  // project the cylinder onto the full-scale scanner proj data
  auto full_size_model_sino_sptr = this->create_data(proj_data_info, emission_map, "full_size_sino_transaxial_block");

  // also project down-sampled, and upsample
  emission_map = VoxelsOnCartesianGrid<float>(this->exam_info_sptr, *downsampled_proj_data_info, 1);
  cyl_map = VoxelsOnCartesianGrid<float>(this->exam_info_sptr, *downsampled_proj_data_info, 1);
  cylinder.construct_volume(cyl_map, CartesianCoordinate3D<int>(1, 1, 1));
  box.construct_volume(emission_map, CartesianCoordinate3D<int>(1, 1, 1));
  emission_map += cyl_map;
  auto interpolated_proj_data_sptr
      = this->create_upsampled_data(proj_data_info, downsampled_proj_data_info, emission_map, "transaxial_block");

  // compare to ground truth
  compare_segment_shape(
      full_size_model_sino_sptr->get_segment_by_sinogram(0), interpolated_proj_data_sptr->get_segment_by_sinogram(0), 3);
}

void
InterpolationTests::transaxial_upsampling_interpolation_test_blocks()
{
  info("Performing transaxial downampled interpolation test for BlocksOnCylindrical scanner");

  // define the original scanner and a downsampled one, as it would be used for scatter simulation
  auto scanner = Scanner(Scanner::User_defined_scanner,
                         "Some_BlocksOnCylindrical_Scanner",
                         96,
                         3,
                         60,
                         60,
                         127,
                         6.5,
                         3.313,
                         1.65,
                         -3.1091819,
                         1,
                         3,
                         3,
                         4,
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
                         7.0,
                         20.0,
                         29.0);
  auto proj_data_info = shared_ptr<ProjDataInfo>(
      std::move(ProjDataInfo::construct_proj_data_info(std::make_shared<Scanner>(scanner), 1, 0, 48, 60, false)));

  // use the code in scatter simulation to downsample the scanner
  auto scatter_simulation = SingleScatterSimulation();
  scatter_simulation.set_template_proj_data_info(proj_data_info);
  scatter_simulation.set_exam_info(*this->exam_info_sptr);
  scatter_simulation.downsample_scanner(-1, 96 / 4); // number of detectors per ring reduced by factor of four
  auto downsampled_proj_data_info = scatter_simulation.get_template_proj_data_info_sptr();

  // define a cylinder precisely in the middle of the FOV
  auto emission_map = VoxelsOnCartesianGrid<float>(this->exam_info_sptr, *downsampled_proj_data_info, 1);
  make_symmetric_object(emission_map);

  // project the cylinder onto the full-scale scanner proj data
  auto full_size_model_sino_sptr = this->create_data(proj_data_info, emission_map, "full_size_sino_transaxial_block_for_LOR");

  // also project down-sampled, and upsample
  emission_map = VoxelsOnCartesianGrid<float>(this->exam_info_sptr, *downsampled_proj_data_info, 1);
  make_symmetric_object(emission_map);
  auto downsampled_proj_data_sptr
      = this->create_data(downsampled_proj_data_info, emission_map, "downsampled_transaxial_block_for_LOR");

  // Identify the bins which should be identical between the downsampled and the interpolated sinogram:
  // Each module has 96 / 8 = 12 crystal in the full size scanner, organised in 3 blocks of 4 crystals, while
  // the downsampled scanner has 3 crystals per module. The idea is that the centre of the outer two
  // is in exactly the same position than the centre of the first and last crystal in the full size scanner.
  SegmentBySinogram<float> sinogram_downsampled = downsampled_proj_data_sptr->get_empty_segment_by_sinogram(0, false, 0);
  SegmentBySinogram<float> sinogram_full_size = full_size_model_sino_sptr->get_empty_segment_by_sinogram(0, false, 0);
  const auto pdi_downsampled = dynamic_cast<const ProjDataInfoGenericNoArcCorr*>(downsampled_proj_data_info.get());
  const auto pdi_full_size = dynamic_cast<const ProjDataInfoGenericNoArcCorr*>(proj_data_info.get());

  int tested_LORs = 0;
  for (int det1_downsampled = 0; det1_downsampled < 3 * 8; det1_downsampled++)
    {
      if (det1_downsampled % 3 == 1)
        continue; // skip the central crystal of each module
      for (int det2_downsampled = 0; det2_downsampled < 3 * 8; det2_downsampled++)
        {
          if (det2_downsampled % 3 == 1 || det1_downsampled == det2_downsampled)
            continue; // skip the central crystal of each module
          if (det1_downsampled / 3 == det2_downsampled / 3)
            continue; // skip the LORs that lie on the same module

          int view_ds, tang_pos_ds;
          pdi_downsampled->get_view_tangential_pos_num_for_det_num_pair(view_ds, tang_pos_ds, det1_downsampled, det2_downsampled);
          BasicCoordinate<3, int> index_downsampled;
          index_downsampled[1] = 1; // looking at central slice
          index_downsampled[2] = view_ds;
          index_downsampled[3] = tang_pos_ds;

          if (tang_pos_ds < pdi_downsampled->get_min_tangential_pos_num()
              || tang_pos_ds > pdi_downsampled->get_max_tangential_pos_num())
            continue;

          int view_fs, tang_pos_fs;
          pdi_full_size->get_view_tangential_pos_num_for_det_num_pair(
              view_fs,
              tang_pos_fs,
              (det1_downsampled / 3) * 12 + ((det1_downsampled % 3) / 2) * 11,
              (det2_downsampled / 3) * 12 + ((det2_downsampled % 3) / 2) * 11);

          BasicCoordinate<3, int> index_full_size;
          index_full_size[1] = 1; // looking at central slice
          index_full_size[2] = view_fs;
          index_full_size[3] = tang_pos_fs;

          if (tang_pos_fs < pdi_full_size->get_min_tangential_pos_num()
              || tang_pos_fs > pdi_full_size->get_max_tangential_pos_num())
            continue;

          // confirm that the difference is smaller than an empirically found value
          check_if_less(std::abs(sinogram_downsampled[index_downsampled] - sinogram_full_size[index_full_size]),
                        0.01,
                        "difference between sinogram bin is larger than expected");

          tested_LORs++;
        }
    }

  info(format("A total of {} LORs were compared between the downsampled and the interpolated sinogram.", tested_LORs));
}

void
InterpolationTests::run_tests()
{
  create_exam_info();
  swap_segments_test();
  extend_projdata_test();
  scatter_interpolation_test_blocks();
  scatter_interpolation_test_cyl();
  scatter_interpolation_test_cyl(true); // 3D
  scatter_interpolation_test_blocks_asymmetric();
  scatter_interpolation_test_cyl_asymmetric();
  scatter_interpolation_test_cyl_asymmetric(true); // 3D
  scatter_interpolation_test_blocks_downsampled();
  transaxial_upsampling_interpolation_test_blocks();
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
