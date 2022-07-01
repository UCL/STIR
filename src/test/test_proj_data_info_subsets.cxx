/*
    Copyright (C) 2021-2022, Commonwealth Scientific and Industrial Research Organisation
    Copyright (C) 2021-2022, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup test
  \ingroup projdata
  \brief Test program for subsetting stir::ProjDataInfo via stir::ProjDataInfoSubsetByView
  \author Ashley Gillman
  \author Kris Thielemans
*/

#include "stir/RunTests.h"
#include "stir/Verbosity.h"
#include "stir/num_threads.h"
#include "stir/CPUTimer.h"
#include "stir/recon_buildblock/ProjectorByBinPairUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/ForwardProjectorByBinUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#include "stir/ProjDataInMemory.h"
#include "stir/Viewgram.h"
#include "stir/Scanner.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/ProjDataInfoSubsetByView.h"
#include "stir/Shape/Ellipsoid.h"
#include <string>

using std::endl;
using std::cerr;
START_NAMESPACE_STIR

std::vector<int>
_calc_regularly_sampled_views_for_subset(int subset_n, int num_subsets, int num_views)
{
  // create a vector containg every num_subsest-th view starting at subset
  // for num_subsets = 4 and subset_n = 0 this is [0, 4, 8, 12, ...]
  // for num_subsets = 4 and subset_n = 1 this is [1, 5, 9, 13, ...]
  std::vector<int> subset_views;
  int view_n = subset_n;

  while (view_n < num_views)
    {
      subset_views.push_back(view_n);
      view_n += num_subsets;
    }

  return subset_views;
}

/*!
  \ingroup test
  \brief Test class for subsets in ProjDataInfo
*/
class TestProjDataInfoSubsets : public RunTests
{
public:
  //! Constructor that can take some input data to run the test with
  TestProjDataInfoSubsets(const std::string& sinogram_filename);

  virtual ~TestProjDataInfoSubsets() {}

  void run_tests();

  void test_split(const ProjData& proj_data);
  // void test_split_and_combine(const ProjData &proj_data, int num_subsets=2);
  void test_forward_projection_is_consistent(const shared_ptr<const VoxelsOnCartesianGrid<float>>& input_image_sptr,
                                             const shared_ptr<const ProjData>& template_sino_sptr, bool use_z_symmetries = false,
                                             bool use_other_symmetries = false, int num_subsets = 10);
  void test_forward_projection_is_consistent_with_unbalanced_subset(
      const shared_ptr<const VoxelsOnCartesianGrid<float>>& input_image_sptr,
      const shared_ptr<const ProjData>& template_sino_sptr, bool use_z_symmetries = false, bool use_other_symmetries = false,
      int num_subsets = 10);
  void test_forward_projection_is_consistent_with_reduced_segment_range(
      const shared_ptr<const VoxelsOnCartesianGrid<float>>& input_image_sptr,
      const shared_ptr<const ProjData>& template_sino_sptr, bool use_z_symmetries = false, bool use_other_symmetries = false,
      int num_subsets = 10);
  void test_back_projection_is_consistent(const shared_ptr<const ProjData>& input_sino_sptr,
                                          const shared_ptr<const VoxelsOnCartesianGrid<float>>& template_image_sptr,
                                          int num_subsets = 10);

protected:
  std::string _sinogram_filename;
  shared_ptr<ProjData> _input_sino_sptr;
  shared_ptr<VoxelsOnCartesianGrid<float>> _test_image_sptr;
  static shared_ptr<VoxelsOnCartesianGrid<float>> construct_test_image_data(const ProjData& template_projdata);
  static shared_ptr<ProjData> construct_test_proj_data();
  static shared_ptr<ProjectorByBinPairUsingProjMatrixByBin>
  construct_projector_pair(const shared_ptr<const ProjDataInfo>& template_projdatainfo_sptr,
                           const shared_ptr<const VoxelsOnCartesianGrid<float>>& template_image_sptr,
                           bool use_z_symmetries = false, bool use_other_symmetries = false);
  static void fill_proj_data_with_forward_projection(const std::shared_ptr<ProjData>& proj_data_sptr,
                                                     const std::shared_ptr<const VoxelsOnCartesianGrid<float>>& test_image_sptr);

  void check_viewgrams(const ProjData& proj_data, const ProjData& subset_proj_data,
                       const std::vector<int>& subset_views,
                       const std::string& str);

  ProjDataInMemory generate_full_forward_projection(const shared_ptr<const VoxelsOnCartesianGrid<float>>& input_image_sptr,
                                                    const shared_ptr<const ProjData>& template_sino_sptr,
                                                    bool use_z_symmetries = false, bool use_other_symmetries = false);
  ProjDataInMemory generate_full_forward_projection(const shared_ptr<const VoxelsOnCartesianGrid<float>>& input_image_sptr,
                                                    const shared_ptr<const ProjDataInfo>& template_projdata_info_sptr,
                                                    const shared_ptr<const ExamInfo>& template_examinfo_sptr,
                                                    bool use_z_symmetries = false, bool use_other_symmetries = false);
  shared_ptr<VoxelsOnCartesianGrid<float>>
  generate_full_back_projection(const shared_ptr<const ProjData>& input_sino_sptr,
                                const shared_ptr<const VoxelsOnCartesianGrid<float>>& template_image_sptr,
                                bool use_z_symmetries = false, bool use_other_symmetries = false);
  void test_forward_projection_for_one_subset(const shared_ptr<const VoxelsOnCartesianGrid<float>>& input_image_sptr,
                                              const ProjDataInMemory& full_forward_projection,
                                              ProjData& subset_forward_projection, bool use_z_symmetries = false,
                                              bool use_other_symmetries = false);
};

TestProjDataInfoSubsets::TestProjDataInfoSubsets(const std::string& sinogram_filename) : _sinogram_filename(sinogram_filename) {}

shared_ptr<ProjData>
TestProjDataInfoSubsets::construct_test_proj_data()
{
  cerr << "\tGenerating default ProjData from E953" << endl;
  shared_ptr<Scanner> scanner_ptr(new Scanner(Scanner::E953));
  shared_ptr<ProjDataInfo> proj_data_info_sptr(
      ProjDataInfo::construct_proj_data_info(scanner_ptr,
                                             /*span*/ 5, scanner_ptr->get_num_rings() - 1,
                                             /*views*/ scanner_ptr->get_num_detectors_per_ring() / 2 / 8,
                                             /*tang_pos*/ 64,
                                             /*arc_corrected*/ false));
  auto exam_info_sptr = std::make_shared<ExamInfo>();
  exam_info_sptr->imaging_modality = ImagingModality::PT;

  return std::make_shared<ProjDataInMemory>(exam_info_sptr, proj_data_info_sptr);
}

shared_ptr<VoxelsOnCartesianGrid<float>>
TestProjDataInfoSubsets::construct_test_image_data(const ProjData& template_projdata)
{
  cerr << "\tGenerating default image of Ellipsoid" << endl;
  auto image = std::make_shared<VoxelsOnCartesianGrid<float>>(template_projdata.get_exam_info_sptr(),
                                                              *template_projdata.get_proj_data_info_sptr());

  // make radius 0.8 FOV
  auto radius = BasicCoordinate<3, float>(image->get_lengths()) * image->get_voxel_size() / 2.F;
  auto centre = image->get_physical_coordinates_for_indices((image->get_min_indices() + image->get_max_indices()) / 2);

  // object at centre of image
  Ellipsoid ellipsoid(radius, centre);

  ellipsoid.construct_volume(*image, Coordinate3D<int>(1, 1, 1));

  cerr << boost::format("\t Generated ellipsoid image, min=%f, max=%f") % image->find_min() % image->find_max() << endl;
  return image;
}

shared_ptr<ProjectorByBinPairUsingProjMatrixByBin>
TestProjDataInfoSubsets::construct_projector_pair(const shared_ptr<const ProjDataInfo>& template_projdatainfo_sptr,
                                                  const shared_ptr<const VoxelsOnCartesianGrid<float>>& template_image_sptr,
                                                  bool use_z_symmetries, bool use_other_symmetries)
{
  cerr << "\tSetting up default projector pair, ProjectorByBinPairUsingProjMatrixByBin" << endl;
  auto proj_matrix_sptr = std::make_shared<ProjMatrixByBinUsingRayTracing>();
  proj_matrix_sptr->set_do_symmetry_180degrees_min_phi(use_other_symmetries);
  proj_matrix_sptr->set_do_symmetry_90degrees_min_phi(use_other_symmetries);
  proj_matrix_sptr->set_do_symmetry_shift_z(use_z_symmetries);
  proj_matrix_sptr->set_do_symmetry_swap_s(use_other_symmetries);
  proj_matrix_sptr->set_do_symmetry_swap_segment(use_other_symmetries);
  auto proj_pair_sptr = std::make_shared<ProjectorByBinPairUsingProjMatrixByBin>(proj_matrix_sptr);

  proj_pair_sptr->set_up(template_projdatainfo_sptr, template_image_sptr);
  return proj_pair_sptr;
}

void
TestProjDataInfoSubsets::fill_proj_data_with_forward_projection(
    const std::shared_ptr<ProjData>& proj_data_sptr, const std::shared_ptr<const VoxelsOnCartesianGrid<float>>& test_image_sptr)
{
  cerr << "\tFilling ProjData with forward projection" << endl;
  auto forward_projector
      = construct_projector_pair(proj_data_sptr->get_proj_data_info_sptr(), test_image_sptr)->get_forward_projector_sptr();

  forward_projector->set_input(*test_image_sptr);
  forward_projector->forward_project(*proj_data_sptr);
}

void
TestProjDataInfoSubsets::run_tests()
{
  cerr << "-------- Testing ProjDataInfoSubsetByView --------\n";
  try
    {
      // Open sinogram
      if (_sinogram_filename.empty())
        {
          _input_sino_sptr = construct_test_proj_data();
          _test_image_sptr = construct_test_image_data(*_input_sino_sptr);
          fill_proj_data_with_forward_projection(_input_sino_sptr, _test_image_sptr);
        }
      else
        {
          _input_sino_sptr = ProjData::read_from_file(_sinogram_filename);
          _test_image_sptr = construct_test_image_data(*_input_sino_sptr);
        }

      // check get_original_view_nums() on the original data
      {
        auto views = _input_sino_sptr->get_original_view_nums();
        check_if_equal(views[0], _input_sino_sptr->get_min_view_num(), "check get_original_view_nums on non-subset data: first view");
        check_if_equal(views[views.size()-1], _input_sino_sptr->get_max_view_num(), "check get_original_view_nums on non-subset data: last view");
      }

      test_split(*_input_sino_sptr);

      // test_split_and_combine(*_input_sino_sptr);

      test_forward_projection_is_consistent(_test_image_sptr, _input_sino_sptr);
      cerr << "repeat with an 'unusual' number of subsets, 13" << endl;
      test_forward_projection_is_consistent(_test_image_sptr, _input_sino_sptr, /*use_z_symmetries=*/false,
                                            /*use_z_symmetries=*/false, /*num_subsets=*/13);
      cerr << "repeat with z shift symmetries" << endl;
      test_forward_projection_is_consistent(_test_image_sptr, _input_sino_sptr, /*use_z_symmetries=*/true,
                                            /*use_z_symmetries=*/false);
      cerr << "repeat with all symmetries" << endl;
      test_forward_projection_is_consistent(_test_image_sptr, _input_sino_sptr, /*use_z_symmetries=*/true,
                                            /*use_z_symmetries=*/true);

      test_forward_projection_is_consistent_with_unbalanced_subset(_test_image_sptr, _input_sino_sptr);

      test_forward_projection_is_consistent_with_reduced_segment_range(_test_image_sptr, _input_sino_sptr);

      test_back_projection_is_consistent(_input_sino_sptr, _test_image_sptr, /*num_subsets=*/10);
    }
  catch (const std::exception& error)
    {
      std::cerr << "\nHere's the error:\n\t" << error.what() << "\n\n";
      everything_ok = false;
    }
  catch (...)
    {
      everything_ok = false;
    }
}

void
TestProjDataInfoSubsets::check_viewgrams(const ProjData& proj_data,
                                         const ProjData& subset_proj_data,
                                         const std::vector<int>& subset_views,
                                         const std::string& str)
{
  // loop over views in the subset data and compare them against the original "full" data
  for (std::size_t i = 0; i < subset_views.size(); ++i)
    {
      // i runs from 0, 1, ... views_in_subset - 1 and indicates the view number in the subset
      // the corresponding view in the original data is at subset_views[i]

      // loop over all segments to check viewgram for all segments
      for (int segment_num = proj_data.get_min_segment_num(); segment_num < proj_data.get_max_segment_num(); ++segment_num)
        {
          if (!check_if_equal(proj_data.get_viewgram(subset_views[i], segment_num),
                              subset_proj_data.get_viewgram(i, segment_num), str + "Are viewgrams equal?"))
            {
              cerr << "test_split failed: viewgrams weren't equal" << endl;
              break;
            }
          // TODO also compare viewgram metadata
        }
    }
}

void
TestProjDataInfoSubsets::test_split(const ProjData& proj_data)
{
  cerr << "\tTesting ability to split a ProjData into consistent subsets" << endl;
  int num_subsets = 4;

  for (int subset_n = 0; subset_n < num_subsets; ++subset_n)
    {
      auto subset_views = _calc_regularly_sampled_views_for_subset(subset_n, num_subsets, proj_data.get_num_views());

      auto subset_proj_data_uptr = proj_data.get_subset(subset_views);
      auto& subset_proj_data = *subset_proj_data_uptr;

      // check basic sizes
      {
        check_if_equal(proj_data.get_num_views() / num_subsets, subset_proj_data.get_num_views(), "check on get_num_views()");
        check_if_equal(proj_data.get_min_tangential_pos_num(), subset_proj_data.get_min_tangential_pos_num(), "check on get_min_tangential_pos_num()");
        check_if_equal(proj_data.get_max_tangential_pos_num(), subset_proj_data.get_max_tangential_pos_num(), "check on get_max_tangential_pos_num()");
        check_if_equal(proj_data.get_min_segment_num(), subset_proj_data.get_min_segment_num(), "check on get_min_segment_num()");
        check_if_equal(proj_data.get_max_segment_num(), subset_proj_data.get_max_segment_num(), "check on get_max_segment_num()");
        for (int segment_num = proj_data.get_min_segment_num(); segment_num <= proj_data.get_max_segment_num(); ++ segment_num)
          {
            check_if_equal(proj_data.get_min_axial_pos_num(segment_num), subset_proj_data.get_min_axial_pos_num(segment_num),
                           "check on get_min_axial_pos_num() for seg " + std::to_string(segment_num));
            check_if_equal(proj_data.get_max_axial_pos_num(segment_num), subset_proj_data.get_max_axial_pos_num(segment_num),
                           "check on get_max_axial_pos_num() for seg " + std::to_string(segment_num));
          }
      }
      check_viewgrams(proj_data, subset_proj_data, subset_views, "get_subset: ");

      // check if we can make a copy
      {
        ProjDataInMemory a_copy(subset_proj_data);
        check_viewgrams(proj_data, subset_proj_data, subset_views, "a copy: ");
      }
    }

  cerr << "\tTesting ProjDataSubsetByView >= operator" << endl;
  auto full_pdi_sptr = proj_data.get_subset(_calc_regularly_sampled_views_for_subset(0, 1, proj_data.get_num_views()))
                           ->get_proj_data_info_sptr();
  auto sub_a_pdi_sptr = proj_data.get_subset(_calc_regularly_sampled_views_for_subset(0, 2, proj_data.get_num_views()))
                            ->get_proj_data_info_sptr();
  auto sub_b_pdi_sptr = proj_data.get_subset(_calc_regularly_sampled_views_for_subset(1, 2, proj_data.get_num_views()))
                            ->get_proj_data_info_sptr();

  // full == orig
  cerr << "\t\tchecking Full subset should >= original ProjDataInfo" << endl;
  if (!(*full_pdi_sptr >= *proj_data.get_proj_data_info_sptr()))
    {
      cerr << typeid(*full_pdi_sptr).name() << " " << typeid(*sub_a_pdi_sptr).name() << endl;
      cerr << "Failed: Expected full == original" << endl;
      everything_ok = false;
    }
  cerr << "\t\tchecking Full subset should >= Smaller subset" << endl;
  if (!(*full_pdi_sptr >= *sub_a_pdi_sptr))
    {
      cerr << "Failed: Expected full >= subset A" << endl;
      everything_ok = false;
    }
  cerr << "\t\tchecking Independent subsets should not >= one another" << endl;
  if ((*sub_a_pdi_sptr >= *sub_b_pdi_sptr))
    {
      cerr << "Failed: Didn't expect subset A >= subset B" << endl;
      everything_ok = false;
    }
}

// void TestProjDataInfoSubsets::
// test_split_and_combine(const ProjData &proj_data, int num_subsets)
// {
//     StandardSubsetter subsetter = StandardSubsetter(proj_data.get_proj_data_info_sptr(), num_subsets);

//     std::vector<ProjData> subsets;
//     for (int s=0; s++; s<num_subsets) {
//         //ProjData& subset = *proj_data.get_subset(s, num_subsets);
//         // or
//         ProjData& subset = *proj_data.get_subset(subsetter.get_views_for_subset(s));
//         subsets.push_back(subset);
//     }

//     ProjData new_proj_data = ProjData(proj_data);  // how to copy?
//     new_proj_data.fill(0);

//     for (int s=0; s++; s<num_subsets) {
//         new_proj_data.fill_subset(s, num_subsets, subsets[s]);
//     }

//     compare_sinos(proj_data, new_proj_data);
// }

void
TestProjDataInfoSubsets::test_forward_projection_is_consistent(
    const shared_ptr<const VoxelsOnCartesianGrid<float>>& input_image_sptr, const shared_ptr<const ProjData>& template_sino_sptr,
    bool use_z_symmetries, bool use_other_symmetries, int num_subsets)
{
  cerr << "\tTesting Subset forward projection is consistent" << endl;

  auto full_forward_projection
      = generate_full_forward_projection(input_image_sptr, template_sino_sptr, use_z_symmetries, use_other_symmetries);

  // ProjData subset;
  for (int subset_n = 0; subset_n < num_subsets; ++subset_n)
    {
      auto subset_views
          = _calc_regularly_sampled_views_for_subset(subset_n, num_subsets, full_forward_projection.get_num_views());
      auto subset_forward_projection_uptr = full_forward_projection.get_subset(subset_views);

      test_forward_projection_for_one_subset(input_image_sptr, full_forward_projection, *subset_forward_projection_uptr,
                                             use_z_symmetries, use_other_symmetries);
    }
}

void
TestProjDataInfoSubsets::test_forward_projection_is_consistent_with_unbalanced_subset(
    const shared_ptr<const VoxelsOnCartesianGrid<float>>& input_image_sptr, const shared_ptr<const ProjData>& template_sino_sptr,
    bool use_z_symmetries, bool use_other_symmetries, int num_subsets)
{
  cerr << "\tTesting Subset forward projection is consistent with unbalanced subset" << endl;

  if (num_subsets >= template_sino_sptr->get_num_views())
    {
      cerr << "Error: Template provided doesn't have enough views to conduct this test with " << num_subsets << " subsets."
           << std::endl;
      everything_ok = false;
    }

  auto full_forward_projection
      = generate_full_forward_projection(input_image_sptr, template_sino_sptr, use_z_symmetries, use_other_symmetries);

  for (int subset_n = 0; subset_n < num_subsets; ++subset_n)
    {
      // subset 0 to subset num_subsets-2 get 1 the view
      // final subset (num_subsets-1) gets the remainder
      std::vector<int> subset_views;

      if (subset_n < num_subsets - 1)
        {
          subset_views.push_back(subset_n);
        }
      else
        {
          for (int view = num_subsets - 1; view < full_forward_projection.get_num_views(); view++)
            {
              subset_views.push_back(view);
            }
        }
      auto subset_forward_projection_uptr = full_forward_projection.get_subset(subset_views);

      cerr << "\tTesting unbalanced subset " << subset_n << ": views " << subset_views << endl;
      test_forward_projection_for_one_subset(input_image_sptr, full_forward_projection, *subset_forward_projection_uptr,
                                             use_z_symmetries, use_other_symmetries);
    }
}

void
TestProjDataInfoSubsets::test_forward_projection_is_consistent_with_reduced_segment_range(
    const shared_ptr<const VoxelsOnCartesianGrid<float>>& input_image_sptr, const shared_ptr<const ProjData>& template_sino_sptr,
    bool use_z_symmetries, bool use_other_symmetries, int num_subsets)
{
  cerr << "\tTesting Subset forward projection is consistent with reduced segment range" << endl;

  if ((template_sino_sptr->get_min_segment_num() > 0) && (template_sino_sptr->get_max_segment_num() < 1))
    {
      cerr << "Error: Template provided doesn't have enough segments to conduct this test with "
           << template_sino_sptr->get_num_segments() << " segments." << std::endl;
      everything_ok = false;
    }

  // First we make a full forward projection with a reduced segment range
  shared_ptr<ProjDataInfo> reduced_seg_range_pdi_sptr(template_sino_sptr->get_proj_data_info_sptr()->clone());
  reduced_seg_range_pdi_sptr->reduce_segment_range(-1, 1);
  auto full_forward_projection
      = generate_full_forward_projection(input_image_sptr, reduced_seg_range_pdi_sptr, template_sino_sptr->get_exam_info_sptr(),
                                         use_z_symmetries, use_other_symmetries);

  for (int subset_n = 0; subset_n < num_subsets; ++subset_n)
    {
      auto subset_views
          = _calc_regularly_sampled_views_for_subset(subset_n, num_subsets, full_forward_projection.get_num_views());

      // Now we make a subset ProjData, but based on the full segment range
      auto subset_template_projdata_uptr = template_sino_sptr->get_subset(subset_views);
      // and independently reduce the segment range on the subset
      shared_ptr<ProjDataInfo> subset_reduced_seg_range_pdi_sptr(
          subset_template_projdata_uptr->get_proj_data_info_sptr()->clone());
      subset_reduced_seg_range_pdi_sptr->reduce_segment_range(-1, 1);
      ProjDataInMemory subset_forward_projection(subset_template_projdata_uptr->get_exam_info_sptr(),
                                                 subset_reduced_seg_range_pdi_sptr);

      cerr << "\tTesting reduced segment range on subset " << subset_n << endl;
      test_forward_projection_for_one_subset(input_image_sptr, full_forward_projection, subset_forward_projection,
                                             use_z_symmetries, use_other_symmetries);
    }
}

ProjDataInMemory
TestProjDataInfoSubsets::generate_full_forward_projection(const shared_ptr<const VoxelsOnCartesianGrid<float>>& input_image_sptr,
                                                          const shared_ptr<const ProjData>& template_projdata_sptr,
                                                          bool use_z_symmetries, bool use_other_symmetries)
{
  return generate_full_forward_projection(input_image_sptr, template_projdata_sptr->get_proj_data_info_sptr(),
                                          template_projdata_sptr->get_exam_info_sptr(), use_z_symmetries, use_other_symmetries);
}

ProjDataInMemory
TestProjDataInfoSubsets::generate_full_forward_projection(const shared_ptr<const VoxelsOnCartesianGrid<float>>& input_image_sptr,
                                                          const shared_ptr<const ProjDataInfo>& template_projdata_info_sptr,
                                                          const shared_ptr<const ExamInfo>& template_examinfo_sptr,
                                                          bool use_z_symmetries, bool use_other_symmetries)
{
  check(input_image_sptr->find_max() > 0, "forward projection test run with empty image");

  auto fwd_projector_sptr
      = construct_projector_pair(template_projdata_info_sptr, input_image_sptr, use_z_symmetries, use_other_symmetries)
            ->get_forward_projector_sptr();

  ProjDataInMemory full_forward_projection(template_examinfo_sptr, template_projdata_info_sptr);
  fwd_projector_sptr->set_input(*input_image_sptr);
  fwd_projector_sptr->forward_project(full_forward_projection);

  check(full_forward_projection.get_viewgram(0, 0).find_max() > 0, "segment 0, view 0 of reference forward projection is empty");

  return full_forward_projection;
}

shared_ptr<VoxelsOnCartesianGrid<float>>
TestProjDataInfoSubsets::generate_full_back_projection(const shared_ptr<const ProjData>& input_sino_sptr,
                                                       const shared_ptr<const VoxelsOnCartesianGrid<float>>& template_image_sptr,
                                                       bool use_z_symmetries, bool use_other_symmetries)
{
  auto back_projector_sptr = construct_projector_pair(input_sino_sptr->get_proj_data_info_sptr(), template_image_sptr,
                                                      use_z_symmetries, use_other_symmetries)
                                 ->get_back_projector_sptr();

  shared_ptr<VoxelsOnCartesianGrid<float>> full_back_projection_sptr(template_image_sptr->get_empty_copy());
  back_projector_sptr->back_project(*full_back_projection_sptr, *input_sino_sptr);

  check(full_back_projection_sptr->find_max() > 0, "full back projection is not empty");

  return full_back_projection_sptr;
}

void
TestProjDataInfoSubsets::test_forward_projection_for_one_subset(
    const shared_ptr<const VoxelsOnCartesianGrid<float>>& input_image_sptr, const ProjDataInMemory& full_forward_projection,
    ProjData& subset_forward_projection, bool use_z_symmetries, bool use_other_symmetries)
{
  cerr << "\tTesting Subset forward projection is consistent" << endl;
  auto subset_proj_data_info_sptr
      = std::dynamic_pointer_cast<const ProjDataInfoSubsetByView>(subset_forward_projection.get_proj_data_info_sptr());

  auto subset_proj_pair_sptr = construct_projector_pair(subset_forward_projection.get_proj_data_info_sptr(), input_image_sptr,
                                                        use_z_symmetries, use_other_symmetries);

  subset_proj_pair_sptr->get_forward_projector_sptr()->forward_project(subset_forward_projection);

  auto subset_views = subset_proj_data_info_sptr->get_original_view_nums();

  // loop over views in the subset data and compare them against the original "full" data
  for (std::size_t i = 0; i < subset_views.size(); ++i)
    {
      // i runs from 0, 1, ... views_in_subset - 1 and indicates the view number in the subset
      // the corresponding view in the original data is at subset_views[i]

      // loop over all segments to check viewgram for all segments
      for (int segment_num = full_forward_projection.get_min_segment_num();
           segment_num < full_forward_projection.get_max_segment_num(); ++segment_num)
        {
          if (!check_if_equal(full_forward_projection.get_viewgram(subset_views[i], segment_num),
                              subset_forward_projection.get_viewgram(i, segment_num), "Are viewgrams equal?"))
            {
              cerr << "testing forward projection failed: viewgrams weren't equal in subset " << i << endl;
              break;
            }
          // TODO also compare viewgram metadata
        }
    }
}

void
TestProjDataInfoSubsets::test_back_projection_is_consistent(
    const shared_ptr<const ProjData>& input_sino_sptr, const shared_ptr<const VoxelsOnCartesianGrid<float>>& template_image_sptr,
    int num_subsets)
{
  auto full_back_projection_sptr = generate_full_back_projection(input_sino_sptr, template_image_sptr);

  shared_ptr<VoxelsOnCartesianGrid<float>> back_projection_sum_sptr(template_image_sptr->get_empty_copy());

  for (int subset_n = 0; subset_n < num_subsets; ++subset_n)
    {
      auto subset_views = _calc_regularly_sampled_views_for_subset(subset_n, num_subsets, input_sino_sptr->get_num_views());

      auto subset = *input_sino_sptr->get_subset(subset_views);

      auto subset_back_projector_sptr
          = construct_projector_pair(subset.get_proj_data_info_sptr(), template_image_sptr)->get_back_projector_sptr();

      shared_ptr<VoxelsOnCartesianGrid<float>> subset_back_projection_sptr(template_image_sptr->get_empty_copy());
      subset_back_projector_sptr->back_project(*subset_back_projection_sptr, subset);
      (*back_projection_sum_sptr) += *subset_back_projection_sptr;
    }
  check_if_equal(*full_back_projection_sptr, *back_projection_sum_sptr, "Are backprojections equal?");
}

END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int
main(int argc, char** argv)
{
  if (argc > 2)
    {
      std::cerr << "\n\tUsage: " << argv[0] << " [projdata_filename]\n";
      return EXIT_FAILURE;
    }

  set_default_num_threads();
  Verbosity::set(0);

  TestProjDataInfoSubsets test(argc > 1 ? argv[1] : "");
  test.run_tests();

  return test.main_return_value();
}
