/*
    Copyright (C) 2011, Hammersmith Imanet Ltd
    Copyright (C) 2013, 2021, 2024 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup recon_test

  \brief Test program for stir::PoissonLogLikelihoodWithLinearModelForMeanAndProjData

  \par Usage

  <pre>
  test_PoissonLogLikelihoodWithLinearModelForMeanAndProjData [proj_data_filename [ density_filename ] ]
  </pre>
  where the 2 arguments are optional. See the class documentation for more info.

  \author Kris Thielemans
  \author Robert Twyman Skelly
*/

#include "stir/recon_buildblock/test/ObjectiveFunctionTests.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/ProjData.h"
#include "stir/ExamInfo.h"
#include "stir/ProjDataInfo.h"
#include "stir/ProjDataInMemory.h"
#include "stir/SegmentByView.h"
#include "stir/Scanner.h"
#include "stir/DataSymmetriesForViewSegmentNumbers.h"
#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMeanAndProjData.h"
#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#include "stir/recon_buildblock/ProjectorByBinPairUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/BinNormalisationFromProjData.h"
#include "stir/recon_buildblock/TrivialBinNormalisation.h"
//#include "stir/OSMAPOSL/OSMAPOSLReconstruction.h"
#include "stir/recon_buildblock/distributable_main.h"
#include "stir/IO/read_from_file.h"
#include "stir/IO/write_to_file.h"
#include "stir/info.h"
#include "stir/Succeeded.h"
#include "stir/num_threads.h"
#include <boost/random/uniform_01.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <iostream>
#include <memory>

#include "stir/IO/OutputFileFormat.h"
#include "stir/recon_buildblock/distributable_main.h"
START_NAMESPACE_STIR

/*!
  \ingroup test
  \brief Test class for PoissonLogLikelihoodWithLinearModelForMeanAndProjData

  This is a somewhat preliminary implementation of a test that compares the result
  of GeneralisedObjectiveFunction::compute_gradient
  with a numerical gradient computed by using the
  GeneralisedObjectiveFunction::compute_objective_function() function.

  The trouble with this is that compute the gradient voxel by voxel is obviously
  terribly slow. A solution (for the test) would be to compute it only in
  a subset of voxels or so. We'll leave this for later.

  Note that the test only works if the objective function is well-defined. For example,
  if certain projections are non-zero, while the model estimates them to be zero, the
  Poisson objective function is in theory infinite.
  PoissonLogLikelihoodWithLinearModelForMeanAndProjData uses some thresholds to try to
  avoid overflow, but if there are too many of these bins, the total objective
  function will become infinite. The numerical gradient then becomes ill-defined
  (even in voxels that do not contribute to these bins).

*/
class PoissonLogLikelihoodWithLinearModelForMeanAndProjDataTests
    : public ObjectiveFunctionTests<PoissonLogLikelihoodWithLinearModelForMeanAndProjData<DiscretisedDensity<3, float>>,
                                    DiscretisedDensity<3, float>>
{
public:
  //! Constructor that can take some input data to run the test with
  /*! This makes it possible to run the test with your own data. However, beware that
      it is very easy to set up a very long computation. See also the note about
      non-zero measured bins.

      \todo it would be better to parse an objective function. That would allow us to set
      all parameters from the command line.
  */
  PoissonLogLikelihoodWithLinearModelForMeanAndProjDataTests(char const* const proj_data_filename = 0,
                                                             char const* const density_filename = 0);
  void construct_input_data(shared_ptr<target_type>& density_sptr, const bool TOF_or_not);

  void run_tests() override;

protected:
  char const* proj_data_filename;
  char const* density_filename;
  shared_ptr<ProjData> proj_data_sptr;
  shared_ptr<ProjData> mult_proj_data_sptr;
  shared_ptr<ProjData> add_proj_data_sptr;
  shared_ptr<PoissonLogLikelihoodWithLinearModelForMeanAndProjData<target_type>> objective_function_sptr;

  //! run the test
  void run_tests_for_objective_function(objective_function_type& objective_function, target_type& target);

  //! Test the approximate Hessian of the objective function by testing the (x^T Hx > 0) condition
  void test_approximate_Hessian_concavity(objective_function_type& objective_function, target_type& target);
};

PoissonLogLikelihoodWithLinearModelForMeanAndProjDataTests::PoissonLogLikelihoodWithLinearModelForMeanAndProjDataTests(
    char const* proj_data_filename, char const* const density_filename)
    : proj_data_filename(proj_data_filename),
      density_filename(density_filename)
{}

void
PoissonLogLikelihoodWithLinearModelForMeanAndProjDataTests::run_tests_for_objective_function(
    objective_function_type& objective_function, target_type& target)
{
  std::cerr << "----- testing Gradient\n";
  test_gradient("PoissonLLProjData", objective_function, target, 0.01F);

  std::cerr << "----- testing concavity via Hessian-vector product (accumulate_Hessian_times_input)\n";
  test_Hessian_concavity("PoissonLLProjData", objective_function, target);

  std::cerr << "----- testing approximate-Hessian-vector product (accumulate_Hessian_times_input)\n";
  test_approximate_Hessian_concavity(objective_function, target);

  std::cerr << "----- testing Hessian-vector product (accumulate_Hessian_times_input)\n";
  test_Hessian("PoissonLLProjData", objective_function, target, 0.5F);

  if (!this->is_everything_ok())
    {
      std::cerr << "Writing diagnostic files proj_data.hs, mult_proj_data.hs and add_proj_data.hs";

      proj_data_sptr->write_to_file("proj_data.hs");
      mult_proj_data_sptr->write_to_file("mult_proj_data.hs");
      add_proj_data_sptr->write_to_file("add_proj_data.hs");
    }
}

void
PoissonLogLikelihoodWithLinearModelForMeanAndProjDataTests::test_approximate_Hessian_concavity(
    objective_function_type& objective_function, target_type& target)
{
  /// setup images
  shared_ptr<target_type> output(target.get_empty_copy());

  /// Compute H x
  objective_function.add_multiplication_with_approximate_Hessian(*output, target);

  /// Compute dot(x,(H x))
  const float my_sum = std::inner_product(target.begin_all(), target.end_all(), output->begin_all(), 0.F);

  // test for a CONCAVE function
  if (this->check_if_less(my_sum, 0))
    {
      //    info("PASS: Computation of x^T H x = " + std::to_string(my_sum) + " < 0" (approximate-Hessian) and is therefore
      //    concave);
    }
  else
    {
      // print to console the FAILED configuration
      info("FAIL: Computation of x^T H x = " + std::to_string(my_sum) + " > 0 (approximate-Hessian) and is therefore NOT concave"
           + "\n >target image max=" + std::to_string(target.find_max())
           + "\n >target image min=" + std::to_string(target.find_min()));
    }
}

void
PoissonLogLikelihoodWithLinearModelForMeanAndProjDataTests::construct_input_data(shared_ptr<target_type>& density_sptr,
                                                                                 const bool TOF_or_not)
{
  if (this->proj_data_filename == 0)
    {
      shared_ptr<ProjDataInfo> proj_data_info_sptr;
      // construct a small scanner and sinogram
      if (TOF_or_not)
        {
          std::cerr << "------ TOF data ----\n";
          shared_ptr<Scanner> scanner_sptr(new Scanner(Scanner::Discovery690));
          scanner_sptr->set_num_rings(4);
          proj_data_info_sptr = std::move(ProjDataInfo::construct_proj_data_info(scanner_sptr,
                                                                                 /*span=*/3,
                                                                                 /*max_delta=*/2,
                                                                                 /*num_views=*/16,
                                                                                 /*num_tang_poss=*/16,
                                                                                 /* arccorrected=*/false,
                                                                                 /* TOF_mash_factor=*/11));
        }
      else
        {
          std::cerr << "------ non-TOF data ----\n";
          shared_ptr<Scanner> scanner_sptr(new Scanner(Scanner::E953));
          scanner_sptr->set_num_rings(5);
          proj_data_info_sptr.reset(ProjDataInfo::ProjDataInfoCTI(scanner_sptr,
                                                                  /*span=*/3,
                                                                  /*max_delta=*/4,
                                                                  /*num_views=*/16,
                                                                  /*num_tang_poss=*/16));
        }
      shared_ptr<ExamInfo> exam_info_sptr(new ExamInfo(ImagingModality::PT));
      proj_data_sptr.reset(new ProjDataInMemory(exam_info_sptr, proj_data_info_sptr));
      for (int seg_num = proj_data_sptr->get_min_segment_num(); seg_num <= proj_data_sptr->get_max_segment_num(); ++seg_num)
        {
          for (int timing_pos_num = proj_data_sptr->get_min_tof_pos_num();
               timing_pos_num <= proj_data_sptr->get_max_tof_pos_num();
               ++timing_pos_num)
            {
              SegmentByView<float> segment = proj_data_sptr->get_empty_segment_by_view(seg_num, false, timing_pos_num);
              // fill in some crazy values
              float value = 0;
              for (SegmentByView<float>::full_iterator iter = segment.begin_all(); iter != segment.end_all(); ++iter)
                {
                  value = float(fabs((seg_num + .1) * value - 5)); // needs to be positive for Poisson
                  *iter = value;
                }
              proj_data_sptr->set_segment(segment);
            }
        }
    }
  else
    {
      proj_data_sptr = ProjData::read_from_file(this->proj_data_filename);
    }

  if (this->density_filename == 0)
    {
      // construct a small image

      CartesianCoordinate3D<float> origin(0, 0, 0);
      const float zoom = 1.F;

      density_sptr.reset(new VoxelsOnCartesianGrid<float>(
          proj_data_sptr->get_exam_info_sptr(), *proj_data_sptr->get_proj_data_info_sptr(), zoom, origin));
      // fill with random numbers between 0 and 1
      typedef boost::mt19937 base_generator_type;
      // initialize by reproducible seed
      static base_generator_type generator(boost::uint32_t(42));
      static boost::uniform_01<base_generator_type> random01(generator);
      for (target_type::full_iterator iter = density_sptr->begin_all(); iter != density_sptr->end_all(); ++iter)
        *iter = static_cast<float>(random01());
    }
  else
    {
      shared_ptr<target_type> aptr(read_from_file<target_type>(this->density_filename));
      density_sptr = aptr;
    }

  // make odd to avoid difficulties with outer-bin that isn't filled-in when using symmetries
  {
    BasicCoordinate<3, int> min_ind, max_ind;
    if (density_sptr->get_regular_range(min_ind, max_ind))
      {
        for (int d = 2; d <= 3; ++d)
          {
            min_ind[d] = std::min(min_ind[d], -max_ind[d]);
            max_ind[d] = std::max(-min_ind[d], max_ind[d]);
          }
        density_sptr->grow(IndexRange<3>(min_ind, max_ind));
      }
  }

  // multiplicative term
  shared_ptr<BinNormalisation> bin_norm_sptr(new TrivialBinNormalisation());
  {

    mult_proj_data_sptr.reset(new ProjDataInMemory(proj_data_sptr->get_exam_info_sptr(),
                                                   proj_data_sptr->get_proj_data_info_sptr()->create_non_tof_clone()));
    for (int seg_num = proj_data_sptr->get_min_segment_num(); seg_num <= proj_data_sptr->get_max_segment_num(); ++seg_num)
      {
        {
          auto segment = mult_proj_data_sptr->get_empty_segment_by_view(seg_num);
          // fill in some crazy values
          float value = 0;
          for (auto iter = segment.begin_all(); iter != segment.end_all(); ++iter)
            {
              value = float(fabs(seg_num * value - .2)); // needs to be positive for Poisson
              *iter = value;
            }
          segment /= 0.5F * segment.find_max(); // normalise to 2 (to avoid large numbers)
          mult_proj_data_sptr->set_segment(segment);
        }
      }
    bin_norm_sptr.reset(new BinNormalisationFromProjData(mult_proj_data_sptr));
  }

  // additive term
  add_proj_data_sptr.reset(new ProjDataInMemory(proj_data_sptr->get_exam_info_sptr(),
                                                proj_data_sptr->get_proj_data_info_sptr()->create_shared_clone()));
  {
    for (int seg_num = proj_data_sptr->get_min_segment_num(); seg_num <= proj_data_sptr->get_max_segment_num(); ++seg_num)
      {
        for (int timing_pos_num = proj_data_sptr->get_min_tof_pos_num(); timing_pos_num <= proj_data_sptr->get_max_tof_pos_num();
             ++timing_pos_num)
          {
            SegmentByView<float> segment = proj_data_sptr->get_empty_segment_by_view(seg_num, false, timing_pos_num);
            // fill in some crazy values
            float value = 0;
            for (SegmentByView<float>::full_iterator iter = segment.begin_all(); iter != segment.end_all(); ++iter)
              {
                value = float(fabs(seg_num * value - .3)); // needs to be positive for Poisson
                *iter = value;
              }
            segment /= 0.333F * segment.find_max(); // normalise to 3 (to avoid large numbers)
            add_proj_data_sptr->set_segment(segment);
          }
      }
  }

  objective_function_sptr.reset(new PoissonLogLikelihoodWithLinearModelForMeanAndProjData<target_type>);
  PoissonLogLikelihoodWithLinearModelForMeanAndProjData<target_type>& objective_function
      = reinterpret_cast<PoissonLogLikelihoodWithLinearModelForMeanAndProjData<target_type>&>(*objective_function_sptr);
  objective_function.set_proj_data_sptr(proj_data_sptr);
  objective_function.set_use_subset_sensitivities(true);
  shared_ptr<ProjMatrixByBin> proj_matrix_sptr(new ProjMatrixByBinUsingRayTracing());
  shared_ptr<ProjectorByBinPair> proj_pair_sptr(new ProjectorByBinPairUsingProjMatrixByBin(proj_matrix_sptr));
  objective_function.set_projector_pair_sptr(proj_pair_sptr);
  /*
    void set_frame_num(const int);
    void set_frame_definitions(const TimeFrameDefinitions&);
  */
  objective_function.set_normalisation_sptr(bin_norm_sptr);
  objective_function.set_additive_proj_data_sptr(add_proj_data_sptr);
  objective_function.set_num_subsets(proj_data_sptr->get_num_views() / 2);
  if (!check(objective_function.set_up(density_sptr) == Succeeded::yes, "set-up of objective function"))
    return;
}

void
PoissonLogLikelihoodWithLinearModelForMeanAndProjDataTests::run_tests()
{
  std::cerr << "Tests for PoissonLogLikelihoodWithLinearModelForMeanAndProjData\n";
  const int verbosity_default = Verbosity::get();
  Verbosity::set(0);

#if 1
  {
    shared_ptr<target_type> density_sptr;
    construct_input_data(density_sptr, /*TOF_or_not=*/false);
    this->run_tests_for_objective_function(*this->objective_function_sptr, *density_sptr);
  }
  if (this->proj_data_filename == 0)
    {
      shared_ptr<target_type> density_sptr;
      construct_input_data(density_sptr, /*TOF_or_not=*/true);
      this->run_tests_for_objective_function(*this->objective_function_sptr, *density_sptr);
    }

#else
  // alternative that gets the objective function from an OSMAPOSL .par file
  // currently disabled
  OSMAPOSLReconstruction<target_type> recon(proj_data_filename); // actually .par
  shared_ptr<GeneralisedObjectiveFunction<target_type>> objective_function_sptr = recon.get_objective_function_sptr();
  if (!check(objective_function_sptr->set_up(recon.get_initial_data_ptr()) == Succeeded::yes, "set-up of objective function"))
    return;
  this->run_tests_for_objective_function(*objective_function_sptr, *recon.get_initial_data_ptr());
#endif
  Verbosity::set(verbosity_default);
}

END_NAMESPACE_STIR

USING_NAMESPACE_STIR

#ifdef STIR_MPI
int
stir::distributable_main(int argc, char** argv)
#else
int
main(int argc, char** argv)
#endif
{
  set_default_num_threads();

  PoissonLogLikelihoodWithLinearModelForMeanAndProjDataTests tests(argc > 1 ? argv[1] : 0, argc > 2 ? argv[2] : 0);
  tests.run_tests();
  return tests.main_return_value();
}
