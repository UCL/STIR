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
#include "stir/zoom.h"
#include "stir/IndexRange3D.h"
#include "stir/Bin.h"
#include "stir/BasicCoordinate.h"
#include "stir/Coordinate3D.h"
#include "stir/recon_buildblock/ProjectorByBinPairUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/ForwardProjectorByBinUsingProjMatrixByBin.h"
/* include SPECTUB matrix*/
#include "stir/recon_buildblock/ProjMatrixByBinSPECTUB.h"
#include "stir/ProjDataInMemory.h"
#include "stir/Viewgram.h"
#include "stir/Scanner.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/ProjDataInfoSubsetByView.h"
#include "stir/Shape/Ellipsoid.h"
#include <stdexcept>
#include <sstream>
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
  explicit TestProjDataInfoSubsets(const std::string& sinogram_filename);

  void run_tests() override;
  void run_tests(const std::shared_ptr<ProjData>& proj_data_sptr,
                 const std::shared_ptr<const VoxelsOnCartesianGrid<float>>& test_image_sptr);

  void test_split(const ProjData& proj_data);
  // void test_split_and_combine(const ProjData &proj_data, int num_subsets=2);
  void test_forward_projection_is_consistent(const shared_ptr<const VoxelsOnCartesianGrid<float>>& input_image_sptr,
                                             const shared_ptr<const ProjData>& template_sino_sptr,
                                             int num_subsets = 10);
  void test_forward_projection_is_consistent_with_unbalanced_subset(
      const shared_ptr<const VoxelsOnCartesianGrid<float>>& input_image_sptr,
      const shared_ptr<const ProjData>& template_sino_sptr,
      int num_subsets = 10);
  void test_back_projection_is_consistent(const shared_ptr<const ProjData>& input_sino_sptr,
                                          const shared_ptr<const VoxelsOnCartesianGrid<float>>& template_image_sptr,
                                          int num_subsets = 10);

protected:
  std::string _sinogram_filename;
  static shared_ptr<VoxelsOnCartesianGrid<float>> construct_test_image_data(const ProjData& template_projdata);
  static shared_ptr<ProjectorByBinPairUsingProjMatrixByBin>
  construct_projector_pair(const shared_ptr<const ProjDataInfo>& template_projdatainfo_sptr,
                           const shared_ptr<const VoxelsOnCartesianGrid<float>>& template_image_sptr);
  static void fill_proj_data_with_forward_projection(const std::shared_ptr<ProjData>& proj_data_sptr,
                                                     const std::shared_ptr<const VoxelsOnCartesianGrid<float>>& test_image_sptr);

  void check_viewgrams(const ProjData& proj_data,
                       const ProjData& subset_proj_data,
                       const std::vector<int>& subset_views,
                       const std::string& str);

  ProjDataInMemory generate_full_forward_projection(const shared_ptr<const VoxelsOnCartesianGrid<float>>& input_image_sptr,
                                                    const shared_ptr<const ProjData>& template_sino_sptr);
  ProjDataInMemory generate_full_forward_projection(const shared_ptr<const VoxelsOnCartesianGrid<float>>& input_image_sptr,
                                                    const shared_ptr<const ProjDataInfo>& template_projdata_info_sptr,
                                                    const shared_ptr<const ExamInfo>& template_examinfo_sptr);
  shared_ptr<VoxelsOnCartesianGrid<float>>
  generate_full_back_projection(const shared_ptr<const ProjData>& input_sino_sptr,
                                const shared_ptr<const VoxelsOnCartesianGrid<float>>& template_image_sptr);
  void test_forward_projection_for_one_subset(const shared_ptr<const VoxelsOnCartesianGrid<float>>& input_image_sptr,
                                              const ProjDataInMemory& full_forward_projection,
                                              ProjData& subset_forward_projection);
};

TestProjDataInfoSubsets::TestProjDataInfoSubsets(const std::string& sinogram_filename)
    : _sinogram_filename(sinogram_filename)
{
  if (_sinogram_filename.empty())
    {
      throw std::runtime_error("Error: No sinogram file provided. Please specify a valid file.");
    }
}

shared_ptr<VoxelsOnCartesianGrid<float>>
TestProjDataInfoSubsets::construct_test_image_data(const ProjData& template_projdata)
{
  cerr << "\tGenerating default image of Ellipsoid" << endl;

  // Get the number of axial positions (slices) from the projection data
  int z_size = template_projdata.get_proj_data_info_sptr()->get_num_axial_poss(0);

  // Get the voxel size and origin from the ProjDataInfo
  auto proj_data_info_sptr = template_projdata.get_proj_data_info_sptr();
  CartesianCoordinate3D<float> voxel_sizes(proj_data_info_sptr->get_sampling_in_t(Bin(0, 0, 0, 0)), // Axial direction
                                           proj_data_info_sptr->get_sampling_in_s(Bin(0, 0, 0, 0)), // Transaxial direction
                                           proj_data_info_sptr->get_sampling_in_s(Bin(0, 0, 0, 0))  // Transaxial direction
  );

  // Use default y_size and x_size from the ProjDataInfo
  int y_size = proj_data_info_sptr->get_num_tangential_poss();
  int x_size = y_size; // Assume square voxels (default for most cases)

  // Define origin
  CartesianCoordinate3D<float> origin(0.0F, 0.0F, 0.0F);

  // Create the 3D index range for the image
  IndexRange3D range(0, z_size - 1, -(y_size / 2), -(y_size / 2) + y_size - 1, -(x_size / 2), -(x_size / 2) + x_size - 1);

  // Create the VoxelsOnCartesianGrid object
  auto image = std::make_shared<VoxelsOnCartesianGrid<float>>(template_projdata.get_exam_info_sptr(), range, origin, voxel_sizes);

  // Make radius 0.8 FOV
  auto radius = BasicCoordinate<3, float>(image->get_lengths()) * image->get_voxel_size() / 2.F;
  auto centre = image->get_physical_coordinates_for_indices((image->get_min_indices() + image->get_max_indices()) / 2);

  // Create an ellipsoid object at the centre of the image
  Ellipsoid ellipsoid(radius, centre);

  // Construct the ellipsoid volume in the image
  ellipsoid.construct_volume(*image, Coordinate3D<int>(1, 1, 1));

  return image;
}

shared_ptr<ProjectorByBinPairUsingProjMatrixByBin>
TestProjDataInfoSubsets::construct_projector_pair(const shared_ptr<const ProjDataInfo>& template_projdatainfo_sptr,
                                                  const shared_ptr<const VoxelsOnCartesianGrid<float>>& template_image_sptr)
{
  cerr << "\tSetting up projector pair with ProjMatrixByBinSPECTUB" << endl;

  // Create the SPECT UB projection matrix
  auto proj_matrix_sptr = std::make_shared<ProjMatrixByBinSPECTUB>();

  // Configure the projection matrix (you can adjust these parameters as needed)
  proj_matrix_sptr->set_attenuation_type("no");              // No attenuation correction by default
  proj_matrix_sptr->set_resolution_model(0.1f, 0.0f, false); // Example resolution model
  proj_matrix_sptr->set_keep_all_views_in_cache(true);       // Optional: optimize for single-threaded computation

  // Set up the projector pair using the SPECT UB projection matrix
  auto proj_pair_sptr = std::make_shared<ProjectorByBinPairUsingProjMatrixByBin>(proj_matrix_sptr);

  // Ensure the projector is initialized with the template projection and image info
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
TestProjDataInfoSubsets::run_tests(const std::shared_ptr<ProjData>& input_sino_sptr,
                                   const std::shared_ptr<const VoxelsOnCartesianGrid<float>>& test_image_sptr)
{
  try
    {

      // check get_original_view_nums() on the original data
      {
        auto views = input_sino_sptr->get_original_view_nums();
        check_if_equal(
            views[0], input_sino_sptr->get_min_view_num(), "check get_original_view_nums on non-subset data: first view");
        check_if_equal(views[views.size() - 1],
                       input_sino_sptr->get_max_view_num(),
                       "check get_original_view_nums on non-subset data: last view");
      }

      test_split(*input_sino_sptr);

      // test_split_and_combine(*input_sino_sptr);

      test_forward_projection_is_consistent(test_image_sptr, input_sino_sptr);
      cerr << "repeat with an 'unusual' number of subsets, 13" << endl;
      test_forward_projection_is_consistent(test_image_sptr,
                                            input_sino_sptr,
                                            /*num_subsets=*/13);

      test_forward_projection_is_consistent_with_unbalanced_subset(test_image_sptr, input_sino_sptr);

      test_back_projection_is_consistent(input_sino_sptr, test_image_sptr, /*num_subsets=*/10);
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
  // Loop over views in the subset data and compare them against the original "full" data
  for (std::size_t i = 0; i < subset_views.size(); ++i)
    {
      // i runs from 0, 1, ... views_in_subset - 1 and indicates the view number in the subset
      // The corresponding view in the original data is at subset_views[i]

      // Loop over all segments to check viewgram for all segments
      for (int segment_num = proj_data.get_min_segment_num(); segment_num <= proj_data.get_max_segment_num(); ++segment_num)
        {
          if (!check_if_equal(proj_data.get_viewgram(subset_views[i], segment_num, false),
                              subset_proj_data.get_viewgram(i, segment_num, false),
                              "Are viewgrams equal?"))
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
        check_if_equal(proj_data.get_min_tangential_pos_num(),
                       subset_proj_data.get_min_tangential_pos_num(),
                       "check on get_min_tangential_pos_num()");
        check_if_equal(proj_data.get_max_tangential_pos_num(),
                       subset_proj_data.get_max_tangential_pos_num(),
                       "check on get_max_tangential_pos_num()");
        check_if_equal(proj_data.get_min_segment_num(), subset_proj_data.get_min_segment_num(), "check on get_min_segment_num()");
        check_if_equal(proj_data.get_max_segment_num(), subset_proj_data.get_max_segment_num(), "check on get_max_segment_num()");
        for (int segment_num = proj_data.get_min_segment_num(); segment_num <= proj_data.get_max_segment_num(); ++segment_num)
          {
            check_if_equal(proj_data.get_min_axial_pos_num(segment_num),
                           subset_proj_data.get_min_axial_pos_num(segment_num),
                           "check on get_min_axial_pos_num() for seg " + std::to_string(segment_num));
            check_if_equal(proj_data.get_max_axial_pos_num(segment_num),
                           subset_proj_data.get_max_axial_pos_num(segment_num),
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
    const shared_ptr<const VoxelsOnCartesianGrid<float>>& input_image_sptr,
    const shared_ptr<const ProjData>& template_sino_sptr,
    int num_subsets)
{
  cerr << "\tTesting Subset forward projection is consistent" << endl;

  auto full_forward_projection = generate_full_forward_projection(input_image_sptr, template_sino_sptr);

  // ProjData subset;
  for (int subset_n = 0; subset_n < num_subsets; ++subset_n)
    {
      auto subset_views
          = _calc_regularly_sampled_views_for_subset(subset_n, num_subsets, full_forward_projection.get_num_views());
      auto subset_forward_projection_uptr = full_forward_projection.get_subset(subset_views);

      test_forward_projection_for_one_subset(input_image_sptr, full_forward_projection, *subset_forward_projection_uptr);
    }
}

void
TestProjDataInfoSubsets::test_forward_projection_is_consistent_with_unbalanced_subset(
    const shared_ptr<const VoxelsOnCartesianGrid<float>>& input_image_sptr,
    const shared_ptr<const ProjData>& template_sino_sptr,
    int num_subsets)
{
  cerr << "\tTesting Subset forward projection is consistent with unbalanced subset" << endl;

  if (num_subsets >= template_sino_sptr->get_num_views())
    {
      cerr << "Error: Template provided doesn't have enough views to conduct this test with " << num_subsets << " subsets."
           << std::endl;
      everything_ok = false;
    }

  auto full_forward_projection = generate_full_forward_projection(input_image_sptr, template_sino_sptr);

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

      std::ostringstream subset_views_msg;
      subset_views_msg << '[';
      for (std::size_t view_idx = 0; view_idx < subset_views.size(); ++view_idx)
        {
          if (view_idx != 0)
            subset_views_msg << ", ";
          subset_views_msg << subset_views[view_idx];
        }
      subset_views_msg << ']';
      cerr << "\tTesting unbalanced subset " << subset_n << ": views " << subset_views_msg.str() << endl;
      test_forward_projection_for_one_subset(input_image_sptr, full_forward_projection, *subset_forward_projection_uptr);
    }
}

ProjDataInMemory
TestProjDataInfoSubsets::generate_full_forward_projection(const shared_ptr<const VoxelsOnCartesianGrid<float>>& input_image_sptr,
                                                          const shared_ptr<const ProjData>& template_projdata_sptr)
{
  return generate_full_forward_projection(
      input_image_sptr, template_projdata_sptr->get_proj_data_info_sptr(), template_projdata_sptr->get_exam_info_sptr());
}

ProjDataInMemory
TestProjDataInfoSubsets::generate_full_forward_projection(const shared_ptr<const VoxelsOnCartesianGrid<float>>& input_image_sptr,
                                                          const shared_ptr<const ProjDataInfo>& template_projdata_info_sptr,
                                                          const shared_ptr<const ExamInfo>& template_examinfo_sptr)
{
  check(input_image_sptr->find_max() > 0, "forward projection test run with empty image");

  auto fwd_projector_sptr = construct_projector_pair(template_projdata_info_sptr, input_image_sptr)->get_forward_projector_sptr();

  ProjDataInMemory full_forward_projection(template_examinfo_sptr, template_projdata_info_sptr);
  fwd_projector_sptr->set_input(*input_image_sptr);
  fwd_projector_sptr->forward_project(full_forward_projection);

  check(full_forward_projection.get_viewgram(0, 0).find_max() > 0, "segment 0, view 0 of reference forward projection is empty");

  return full_forward_projection;
}

shared_ptr<VoxelsOnCartesianGrid<float>>
TestProjDataInfoSubsets::generate_full_back_projection(const shared_ptr<const ProjData>& input_sino_sptr,
                                                       const shared_ptr<const VoxelsOnCartesianGrid<float>>& template_image_sptr)
{
  auto back_projector_sptr
      = construct_projector_pair(input_sino_sptr->get_proj_data_info_sptr(), template_image_sptr)->get_back_projector_sptr();

  shared_ptr<VoxelsOnCartesianGrid<float>> full_back_projection_sptr(template_image_sptr->get_empty_copy());
  back_projector_sptr->back_project(*full_back_projection_sptr, *input_sino_sptr);

  check(full_back_projection_sptr->find_max() > 0, "full back projection is not empty");

  return full_back_projection_sptr;
}

void
TestProjDataInfoSubsets::test_forward_projection_for_one_subset(
    const shared_ptr<const VoxelsOnCartesianGrid<float>>& input_image_sptr,
    const ProjDataInMemory& full_forward_projection,
    ProjData& subset_forward_projection)
{
  cerr << "\tTesting Subset forward projection is consistent" << endl;
  auto subset_proj_data_info_sptr
      = std::dynamic_pointer_cast<const ProjDataInfoSubsetByView>(subset_forward_projection.get_proj_data_info_sptr());

  auto subset_proj_pair_sptr = construct_projector_pair(subset_forward_projection.get_proj_data_info_sptr(), input_image_sptr);

  subset_proj_pair_sptr->get_forward_projector_sptr()->forward_project(subset_forward_projection);

  auto subset_views = subset_proj_data_info_sptr->get_original_view_nums();

  // loop over views in the subset data and compare them against the original "full" data
  for (std::size_t i = 0; i < subset_views.size(); ++i)
    {
      // i runs from 0, 1, ... views_in_subset - 1 and indicates the view number in the subset
      // the corresponding view in the original data is at subset_views[i]

      // loop over all segments to check viewgram for all segments
      for (int segment_num = full_forward_projection.get_min_segment_num();
           segment_num < full_forward_projection.get_max_segment_num();
           ++segment_num)
        {
          if (!check_if_equal(full_forward_projection.get_viewgram(subset_views[i], segment_num, false),
                              subset_forward_projection.get_viewgram(i, segment_num, false),
                              "Are viewgrams equal?"))
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
    const shared_ptr<const ProjData>& input_sino_sptr,
    const shared_ptr<const VoxelsOnCartesianGrid<float>>& template_image_sptr,
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

void
TestProjDataInfoSubsets::run_tests()
{
  cerr << "-------- Testing ProjDataInfoSubsetByView --------\n";

  // Step 1: Construct the test image
  cerr << "\tConstructing test image...\n";
  auto input_sino_sptr = ProjData::read_from_file(_sinogram_filename);
  auto test_image_sptr = construct_test_image_data(*input_sino_sptr);

  // Step 2: Generate forward projection from the test image
  cerr << "\tGenerating forward projection of the test image...\n";
  auto generated_sino_sptr
      = std::make_shared<ProjDataInMemory>(input_sino_sptr->get_exam_info_sptr(), input_sino_sptr->get_proj_data_info_sptr());

  fill_proj_data_with_forward_projection(generated_sino_sptr, test_image_sptr);

  // Step 3: Run tests using the generated forward projection
  cerr << "\tRunning tests with the generated forward projection...\n";
  run_tests(generated_sino_sptr, test_image_sptr);
}

END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int
main(int argc, char** argv)
{
  std::cerr << "SPECT Executable built and running successfully!" << std::endl;
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
