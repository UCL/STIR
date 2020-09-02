/*
    Copyright (C) 2020, CSIRO
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup test
  
  \brief Test program for various parts of code that map sinogram and
  gantry space back to image space. Make sure they're consistent and
  consider gantry coordinate (0, 0, 0) to be the centre of the gantry,
  regardless of image location.

  Tests:
    - stir::ProjMatrixByBinUsingRayTracing.
    - (TODO) stir::ProjMatrixByBinUsingInterpolation.

  Notable omissions:
    - DataSymmetriesForBins performs mapping. It is tested under the
      data symmetries test against RT projector.
    
   \author Ashley Gillman
      
*/

#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/IndexRange.h"
#include "stir/IndexRange3D.h"
#include "stir/ProjDataInfo.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/ProjData.h"
#include "stir/ProjDataInMemory.h"
#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#include "stir/recon_buildblock/ProjMatrixByBinUsingInterpolation.h"
#include "stir/recon_buildblock/ProjectorByBinPair.h"
#include "stir/recon_buildblock/ProjectorByBinPairUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/ProjectorByBinPairUsingSeparateProjectors.h"
// #include "stir/recon_buildblock/ForwardProjectorByBinUsingRayTracing.h"
// #include "stir/recon_buildblock/ForwardProjectorByBinUsingInterpolation.h"
// #include "stir/recon_buildblock/BackProjectorByBinUsingRayTracing.h"
// #include "stir/recon_buildblock/BackProjectorByBinUsingInterpolation.h"
#include "stir/Shape/EllipsoidalCylinder.h"
#include "stir/SeparableGaussianImageFilter.h"
#include "stir/IO/write_to_file.h"
#include "stir/RunTests.h"
#include "stir/info.h"
#include "stir/stream.h"
#include <iostream>
#include <sstream>
#include <math.h>
#ifndef STIR_NO_NAMESPACES
using std::stringstream;
#endif

START_NAMESPACE_STIR

struct ForwardBackProjectorPair {
  // metadata
  const std::string forward_name;
  const std::string back_name;
  const std::string name;
  const bool matched_projectors;
  const shared_ptr<ForwardProjectorByBin> forward_projector_sptr;
  const shared_ptr<BackProjectorByBin> back_projector_sptr;
  ForwardBackProjectorPair(
    const std::string forward_name, const std::string back_name,
    const bool matched_projectors,
    const shared_ptr<ForwardProjectorByBin> forward_projector_sptr,
    const shared_ptr<BackProjectorByBin> back_projector_sptr,
    const std::string name_="")
    : forward_name{ forward_name }, back_name{ back_name },
      name { forward_name + "-" + back_name },
      matched_projectors{ matched_projectors },
      forward_projector_sptr{ forward_projector_sptr },
      back_projector_sptr{ back_projector_sptr } { };
};

struct OneForwardBackTest {
  // metadata
  const std::string name;

  // test inputs
  const shared_ptr<ProjDataInfo> proj_data_info_sptr;
  const shared_ptr<ExamInfo> exam_info_sptr;
  const shared_ptr<VoxelsOnCartesianGrid<float> > image_sptr;
  const ForwardBackProjectorPair projector_pair;

  // test results
  shared_ptr<ProjDataInMemory> projection_sptr;
  shared_ptr<VoxelsOnCartesianGrid<float> > backprojection_sptr;

  OneForwardBackTest(
    const shared_ptr<ProjDataInfo> proj_data_info_sptr,
    const shared_ptr<ExamInfo> exam_info_sptr,
    const shared_ptr<VoxelsOnCartesianGrid<float> > image_sptr,
    const ForwardBackProjectorPair projector_pair,
    const std::string name="")
    : name{ name + "-" + projector_pair.name },
      proj_data_info_sptr{ proj_data_info_sptr },
      exam_info_sptr{ exam_info_sptr },
      image_sptr{ image_sptr },
      projector_pair{ projector_pair }
  {
    projection_sptr.reset(
      new ProjDataInMemory(exam_info_sptr, proj_data_info_sptr));
    backprojection_sptr.reset(image_sptr->get_empty_copy());
  };
};

class CoordinateSystemMappingTests : public RunTests
{
public:
  CoordinateSystemMappingTests(char const * template_proj_data_filename = 0);

  void run_tests();

  //! check correlations of vectors
  template <class T>
  bool check_if_correlated(
    const VectorWithOffset<T>& t1, const VectorWithOffset<T>& t2,
    const std::string& str="");

  void set_correlation_tolerance(const double correlation_tolerance);
  double get_correlation_tolerance();

private:
  char const * template_proj_data_filename;

  double correlation_tolerance = 0.99;

  Succeeded run_tests_for_1_projdatainfo(
    const shared_ptr<ProjDataInfo> proj_data_info_sptr,
    const shared_ptr<ExamInfo> exam_info_sptr);

  Succeeded run_tests_for_1_projdata_extended_axial_fov(
    const shared_ptr<ProjDataInfo> proj_data_info_sptr,
    const shared_ptr<ExamInfo> exam_info_sptr,
    const shared_ptr<VoxelsOnCartesianGrid<float> > standard_image_sptr,
    const std::vector<ForwardBackProjectorPair>& projector_pairs);

  Succeeded one_forward_backward_test(const OneForwardBackTest& test);

  const float BACKPROJ_TO_ORIG_COR_TOLERANCE = 0.86;
  const float DIFFERENT_PROJECTORS_TOLERANCE = 0.01;
  const float SAME_PROJECTORS_TOLERANCE = 0.001;
};

CoordinateSystemMappingTests::
CoordinateSystemMappingTests(char const * template_proj_data_filename)
  : template_proj_data_filename(template_proj_data_filename)
{}

bool recursive_calculate_correlation(
    float& sum_x, float& sum_y, float& sum_xx, float& sum_yy, float& sum_xy,
    int& n, const float x, const float y) {
  sum_x += x;
  sum_y += y;
  sum_xx += x*x;
  sum_yy += y*y;
  sum_xy += x*y;
  n += 1;
  return true;
}

template <class T>
bool recursive_calculate_correlation(
    float& sum_x, float& sum_y, float& sum_xx, float& sum_yy, float& sum_xy,
    int& n, const VectorWithOffset<T>& x, const VectorWithOffset<T>& y) {
  if (x.get_min_index() != y.get_min_index()
      or x.get_max_index() != y.get_max_index()) {
    return false;
  }
  for (int i = x.get_min_index(); i <= x.get_max_index(); i++) {
    bool still_going = recursive_calculate_correlation(
      sum_x, sum_y, sum_xx, sum_yy, sum_xy, n, x[i], y[i]);
    if (not still_going) { return false; }
  }
  return true;
}

template <class T>
bool CoordinateSystemMappingTests::check_if_correlated(
    const VectorWithOffset<T>& t1, const VectorWithOffset<T>& t2,
    const std::string& str) {
  float sum_x = 0;
  float sum_y = 0;
  float sum_xx = 0;
  float sum_yy = 0;
  float sum_xy = 0;
  int n = 0;
  if (recursive_calculate_correlation(
    sum_x, sum_y, sum_xx, sum_yy, sum_xy, n, t1, t2)) {
    if (n == 0) {
      std::cerr << "Error: vectors are empty. " << str << std::endl;
      return everything_ok = false;
    }
    const float correlation = (n*sum_xy - sum_x*sum_y)
      / sqrt((n*sum_xx - sum_x*sum_x) * (n*sum_yy - sum_y*sum_y));
    if (correlation < correlation_tolerance) {
      std::cerr << "Error: Uncorrelated vectors, correlation was only " 
                << correlation << ". " << str << std::endl;
      return everything_ok = false;
    } else {
      info(boost::format("Correlation: %s") % correlation);
      return true;
    }
  }
  else {
    std::cerr << "Error: unequal ranges: ("
      << t1.get_min_index() << "," << t1.get_max_index() << ") ("
      << t2.get_min_index() << "," << t2.get_max_index() << "). "
      << str << std::endl;
    return everything_ok = false;
  }
}

void CoordinateSystemMappingTests::set_correlation_tolerance(
    const double correlation_tolerance_v) {
  correlation_tolerance = correlation_tolerance_v;
}

void
fill_in_cylinder_test_image(VoxelsOnCartesianGrid<float>& image) {
  const float cyl_offset_x = 10;
  const float cyl_offset_y = 5;
  const float cyl_offset_z = 5;
  const CartesianCoordinate3D<float>
    cyl_orig = CartesianCoordinate3D<float>(cyl_offset_z, cyl_offset_y, cyl_offset_x)
    + image.get_image_centre_in_physical_coordinates();
  const float cyl_length = image.get_z_size() * image.get_voxel_size().z() - cyl_offset_z - 50;
  const float cyl_rad_x = 15;
  const float cyl_rad_y = 20;
  // std::cerr
  //   << "image origin: " << image.get_origin() << std::endl
  //   << "image size:" << image_size
  //   << " = " << image.get_lengths() << " * " << image.get_grid_spacing() << std::endl
  //   << "cylinder centre: " << cyl_orig << std::endl
  //   << "cyl_length: " << cyl_length << std::endl;

  EllipsoidalCylinder cylinder(cyl_length, cyl_rad_y, cyl_rad_x, cyl_orig);
  cylinder.construct_volume(image, CartesianCoordinate3D<int>(2,2,2));

  // filter it a bit to avoid too high frequency stuff creating trouble in the comparison
  const float fwhm = 10;
  SeparableGaussianImageFilter<float> filter;
  filter.set_fwhms(make_coordinate(fwhm, fwhm, fwhm));
  filter.set_up(image);
  filter.apply(image);
}

// Succeeded
// CoordinateSystemMappingTests::
// run_tests_for_1_projdata_extended_axial_fov_for_1_projector_pair(
//   const shared_ptr<ProjDataInfo> proj_data_info_sptr,
//   const shared_ptr<ExamInfo> exam_info_sptr,
//   const shared_ptr<VoxelsOnCartesianGrid<float> > standard_image_sptr,
//   const std::vector<shared_ptr<ProjectorByBinPair> > projector_pairs)
// {

void do_forward_then_back_projections(
  ProjData& projection, shared_ptr<VoxelsOnCartesianGrid<float> > backprojection_sptr,
  const shared_ptr<ProjDataInfo> proj_data_info_sptr,
  const shared_ptr<VoxelsOnCartesianGrid<float> > standard_image_sptr,
  const ForwardBackProjectorPair& projector_pair)
{
    // set up projectors
    projector_pair.forward_projector_sptr->set_up(proj_data_info_sptr, standard_image_sptr);
    projector_pair.back_projector_sptr->set_up(proj_data_info_sptr, standard_image_sptr);
    // shared_ptr<ForwardProjectorByBin> fwd_projector_sptr =
    //   projector_pair->get_forward_projector_sptr();
    // std::cerr << "... using " << fwd_projector_sptr->get_registered_name()
    //           << " for forward:"  << std::endl
    //           << fwd_projector_sptr->parameter_info();
    // shared_ptr<BackProjectorByBin> bck_projector_sptr =
    //   projector_pair->get_back_projector_sptr();
    // std::cerr << "... using " << bck_projector_sptr->get_registered_name()
    //           << " for backward:"  << std::endl
    //           << bck_projector_sptr->parameter_info();

    // std::cerr << "# Test with fwd(" << fwd_projector_sptr->get_registered_name()
    //           << ") bck(" << bck_projector_sptr->get_registered_name() << ")"
    //           << std::endl;
    // std::cerr << std::endl;
    // std::cerr << "## Forward and back project standard image..." << std::endl;

    std::cerr << std::endl;
    std::cerr << "### Running forward" << std::endl;
    projector_pair.forward_projector_sptr->set_input(*standard_image_sptr);
    projector_pair.forward_projector_sptr->forward_project(projection);

    std::cerr << std::endl;
    std::cerr << "### Running backward" << std::endl;
    projector_pair.back_projector_sptr->start_accumulating_in_new_target();
    projector_pair.back_projector_sptr->back_project(projection);
    projector_pair.back_projector_sptr->get_output(*backprojection_sptr);
}

// Forward, back, and check round-trip consistecy by correlation
Succeeded
CoordinateSystemMappingTests::
one_forward_backward_test(const OneForwardBackTest& test) {
    std::cerr << "##" << std::endl;
    std::cerr << "## Testing fwd-bck rountrip " << test.name << std::endl;
    std::cerr << "##" << std::endl;
    std::cerr << std::endl;

    // project
    do_forward_then_back_projections(
      *test.projection_sptr, test.backprojection_sptr,
      test.proj_data_info_sptr, test.image_sptr, test.projector_pair);

    // round-trip test
      if (not check_if_correlated(
          *test.image_sptr, *test.backprojection_sptr,
        "Checking backprojection matches original.")) {
      std::cerr << "Saving last failed image comparison to "
                << test.name << "_failed_{ref,oth}.hv. "
                << "(may be overwritten)" << std::endl;
      write_to_file(test.name + "_failed_ref.hv", *test.image_sptr);
      write_to_file(test.name + "_failed_oth.hv", *test.backprojection_sptr);
      return Succeeded::no;
    }

    return Succeeded::yes;
  }

// Motivated by changing the definition of the coordinate system mapping to/from
// image space and sinogram space, we want to make sure that the desired
// behaviour of having oversize and non-centred-in-z-plane images are consistent
// with the standard image size.
Succeeded
CoordinateSystemMappingTests::run_tests_for_1_projdata_extended_axial_fov(
  const shared_ptr<ProjDataInfo> proj_data_info_sptr,
  const shared_ptr<ExamInfo> exam_info_sptr,
  const shared_ptr<VoxelsOnCartesianGrid<float> > standard_image_sptr,
  const std::vector<ForwardBackProjectorPair>& projector_pairs)
{
  Succeeded succeeded = Succeeded::yes;

  const int EXTEND_BY = 3;

  for (ForwardBackProjectorPair projector_pair : projector_pairs) {
    std::cerr << "#" << std::endl;
    std::cerr << "# Testing " << projector_pair.name << std::endl;
    std::cerr << "#" << std::endl;

    std::cerr << "... test with standard FOV" << std::endl;
    OneForwardBackTest standard_fov_test = OneForwardBackTest(
      proj_data_info_sptr, exam_info_sptr, standard_image_sptr,
      projector_pair, "standard-FOV");

    std::vector<OneForwardBackTest> extended_fov_tests;

    {
      shared_ptr<VoxelsOnCartesianGrid<float> > image_sptr(
        standard_image_sptr->clone());
      image_sptr->grow_z_range(
        standard_image_sptr->get_min_z() - EXTEND_BY,
        standard_image_sptr->get_max_z() + EXTEND_BY);
      extended_fov_tests.push_back(OneForwardBackTest(
        proj_data_info_sptr, exam_info_sptr, image_sptr,
        projector_pair, "extended-FOV-both"));
    }

    {
      std::cerr << "... extend in positive z direction" << std::endl;
      shared_ptr<VoxelsOnCartesianGrid<float> > image_sptr(
        standard_image_sptr->clone());
      image_sptr->grow_z_range(
        standard_image_sptr->get_min_z(),
        standard_image_sptr->get_max_z() + EXTEND_BY);
      extended_fov_tests.push_back(OneForwardBackTest(
        proj_data_info_sptr, exam_info_sptr, image_sptr,
        projector_pair, "extended-FOV-positive"));
    }

    {
      std::cerr << "... extend in negative z direction" << std::endl;
      shared_ptr<VoxelsOnCartesianGrid<float> > image_sptr(
        standard_image_sptr->clone());
      image_sptr->grow_z_range(
        standard_image_sptr->get_min_z() - EXTEND_BY,
        standard_image_sptr->get_max_z());
      extended_fov_tests.push_back(OneForwardBackTest(
        proj_data_info_sptr, exam_info_sptr, image_sptr,
        projector_pair, "extended-FOV-negative"));
    }

    std::cerr << std::endl;

    // Tests

    // first the standard FOV
    succeeded &= one_forward_backward_test(standard_fov_test);

    for (OneForwardBackTest extended_fov_test : extended_fov_tests) {

      // then each extended FOV
      succeeded &= one_forward_backward_test(extended_fov_test);

      // standard/extended FOV projection comparison
      // TODO: which is faster/natural - viewgram or sinogram?
      // These are within 0.01, but I thought they'd actually be closer..
      set_tolerance(SAME_PROJECTORS_TOLERANCE);
      if (not check_if_equal<float>(
          standard_fov_test.projection_sptr->begin_all(),
          standard_fov_test.projection_sptr->end_all(),
          extended_fov_test.projection_sptr->begin_all(),
          extended_fov_test.projection_sptr->end_all(),
          "Checking extended FOV projection matches standard.")) {
        succeeded = Succeeded::no;
        std::cerr
          << "Saving last failed image comparison to "
          << extended_fov_test.name << "_fovdiff_failed_{ref,oth}.hs. "
          << "(may be overwritten)" << std::endl;
        standard_fov_test.projection_sptr->write_to_file(extended_fov_test.name + "_fovdiff_failed_ref.hs");
        extended_fov_test.projection_sptr->write_to_file(extended_fov_test.name + "_fovdiff_failed_oth.hs");
      }
    }

    std::cerr << std::endl;
  }

  // for (
  //   std::vector<VoxelsOnCartesianGrid<float>>::iterator extended_image = extended_image_sptrs.begin();
  //   extended_image != extended_image_sptrs.end();
  //   ++extended_image) {
  //     fwd_projector_sptr->set_up(proj_data_info_sptr, standard_image_sptr);
  //   }
  
  return succeeded;
}


Succeeded
CoordinateSystemMappingTests::run_tests_for_1_projdatainfo(
  const shared_ptr<ProjDataInfo> proj_data_info_sptr,
  const shared_ptr<ExamInfo> exam_info_sptr)
{
  Succeeded succeeded = Succeeded::yes;
  const float zoom = 1.F;

  set_correlation_tolerance(BACKPROJ_TO_ORIG_COR_TOLERANCE);

  CartesianCoordinate3D<float> standard_origin(0,0,0);
  shared_ptr<VoxelsOnCartesianGrid<float> > 
    standard_image_sptr(
      new VoxelsOnCartesianGrid<float>(exam_info_sptr, *proj_data_info_sptr, zoom, standard_origin));
  fill_in_cylinder_test_image(*standard_image_sptr);

  // DiscretisedDensity<3,float> extended_axial_dir_1_image =
  //   standard_density_sptr->

  // Define paired projectors
  // (Do they now share a matrix and save memory?)
  shared_ptr<ProjMatrixByBin> matrix_raytrace_sptr(
    new ProjMatrixByBinUsingRayTracing());
  shared_ptr<ProjectorByBinPairUsingProjMatrixByBin>
    projector_pair_matrix_raytrace_sptr(
      new ProjectorByBinPairUsingProjMatrixByBin(matrix_raytrace_sptr));
  shared_ptr<ProjMatrixByBin> matrix_interp_sptr(
    new ProjMatrixByBinUsingInterpolation());
  shared_ptr<ProjectorByBinPairUsingProjMatrixByBin>
    projector_pair_matrix_interp_sptr(
    new ProjectorByBinPairUsingProjMatrixByBin(matrix_interp_sptr));

  // Define forward projectors
  using NameAndFwdProjPair =
    std::pair<std::string, shared_ptr<ForwardProjectorByBin> >;
  std::vector<NameAndFwdProjPair> fwd_projectors;
  NameAndFwdProjPair reference_fwd_projector = 
    NameAndFwdProjPair(
      "RayTracingMatrix",
      projector_pair_matrix_raytrace_sptr->get_forward_projector_sptr());
  // don't add the RT, all combos already covered
  // fwd_projectors.push_back(
  //   std::pair(
  //     "RayTracingMatrix",
  //     projector_pair_matrix_raytrace_sptr->get_forward_projector_sptr()));
  fwd_projectors.push_back(
    NameAndFwdProjPair(
      "InterpolationMatrix",
      projector_pair_matrix_interp_sptr->get_forward_projector_sptr()));

  // Define backward projectors
  using NameAndBckProjPair =
    std::pair<std::string, shared_ptr<BackProjectorByBin> >;
  std::vector<NameAndBckProjPair> bck_projectors;
  NameAndBckProjPair reference_bck_projector =
    NameAndBckProjPair(
      "RayTracingMatrix",
      projector_pair_matrix_raytrace_sptr->get_back_projector_sptr());
  // don't add the RT, all combos already covered
  // bck_projectors.push_back(
  //   std::pair(
  //     "RayTracingMatrix",
  //     projector_pair_matrix_raytrace_sptr->get_back_projector_sptr()));
  bck_projectors.push_back(
    NameAndBckProjPair(
      "InterpolationMatrix",
      projector_pair_matrix_interp_sptr->get_back_projector_sptr()));

  // Set up pairs to test
  std::vector<ForwardBackProjectorPair> projector_pairs;
  // Test the correctly paired projectors first
  projector_pairs.push_back(ForwardBackProjectorPair(
    "RayTraceMatrix", "RayTraceMatrix", true,
    projector_pair_matrix_raytrace_sptr->get_forward_projector_sptr(),
    projector_pair_matrix_raytrace_sptr->get_back_projector_sptr()));

  std::cerr << "name: " << projector_pairs[0].name << std::endl;
  std::cerr << projector_pair_matrix_raytrace_sptr->get_forward_projector_sptr()->get_registered_name() << std::endl;
  std::cerr << projector_pairs[0].forward_projector_sptr->get_registered_name() << std::endl;

  projector_pairs.push_back(ForwardBackProjectorPair(
    "InterpolationMatrix", "InterpolationMatrix", true,
    projector_pair_matrix_interp_sptr->get_forward_projector_sptr(),
    projector_pair_matrix_interp_sptr->get_back_projector_sptr()));
  // each back projector against the reference forward projector
  for (NameAndBckProjPair bck_projector : bck_projectors) {
    projector_pairs.push_back(
      ForwardBackProjectorPair(
        reference_fwd_projector.first, bck_projector.first, false,
        reference_fwd_projector.second, bck_projector.second));
  }
  // each forward projector against the reference back projector
  for (NameAndFwdProjPair fwd_projector : fwd_projectors) {
    projector_pairs.push_back(
      ForwardBackProjectorPair(
        fwd_projector.first, reference_bck_projector.first, false,
        fwd_projector.second, reference_bck_projector.second));
  }

  // Now test each of the pairs
  succeeded &= run_tests_for_1_projdata_extended_axial_fov(
      proj_data_info_sptr, exam_info_sptr,
      standard_image_sptr, projector_pairs);

  std::cerr << "saving cylinder.hv" << std::endl;
  write_to_file("cylinder.hv", *standard_image_sptr);

  return succeeded;
}

void CoordinateSystemMappingTests::run_tests()
{
  std::cerr << "Tests for DataSymmetriesForBins_PET_CartesianGrid\n";

  shared_ptr<ProjDataInfo> proj_data_info_sptr;
  shared_ptr<ExamInfo> exam_info_sptr(new ExamInfo);
  exam_info_sptr->imaging_modality = ImagingModality::PT;

  if (template_proj_data_filename == 0) {
    {  
      std::cerr << "Testing span=1\n";
      shared_ptr<Scanner> scanner_sptr(new Scanner(Scanner::E953));
      proj_data_info_sptr.reset( 
          ProjDataInfo::ProjDataInfoCTI(scanner_sptr, 
          /*span=*/1, 
          /*max_delta=*/5,
          /*num_views=*/8,
          /*num_tang_poss=*/16));

      run_tests_for_1_projdatainfo(proj_data_info_sptr, exam_info_sptr);
    }
    {  
      std::cerr << "Testing span=3\n";
      // warning: make sure that parameters are ok such that hard-wired
      // bins above are fine (e.g. segment 3 should be allowed)
      shared_ptr<Scanner> scanner_sptr(new Scanner(Scanner::E953));
      proj_data_info_sptr.reset(
      ProjDataInfo::ProjDataInfoCTI(scanner_sptr, 
          /*span=*/3, 
          /*max_delta=*/12,
          /*num_views=*/8,
          /*num_tang_poss=*/16));

      run_tests_for_1_projdatainfo(proj_data_info_sptr, exam_info_sptr);
    }
  } else {
    shared_ptr<ProjData> proj_data_sptr =
      ProjData::read_from_file(template_proj_data_filename);
    proj_data_info_sptr = 
      proj_data_sptr->get_proj_data_info_sptr()->create_shared_clone();

    run_tests_for_1_projdatainfo(proj_data_info_sptr, exam_info_sptr);
  }
}

END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int main(int argc, char **argv)
{
  CoordinateSystemMappingTests tests(argc==2? argv[1] : 0);
  tests.run_tests();
  return tests.main_return_value();
}
