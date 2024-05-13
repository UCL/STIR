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

  \brief Test program for stir::PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin

  \par Usage

  <pre>
  test_PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin lm_data_filename [ density_filename ]
  </pre>
  where the last arguments are optional. See the class documentation for more info.

  \author Kris Thielemans
  \author Robert Twyman Skelly
*/

#include "stir/recon_buildblock/test/ObjectiveFunctionTests.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/listmode/CListModeData.h"
#include "stir/ExamInfo.h"
#include "stir/ProjDataInfo.h"
#include "stir/ProjDataInMemory.h"
#include "stir/SegmentByView.h"
#include "stir/Scanner.h"
#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin.h"
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

START_NAMESPACE_STIR

/*!
  \ingroup test
  \brief Test class for PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin

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
  PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin uses some thresholds to try to
  avoid overflow, but if there are too many of these bins, the total objective
  function will become infinite. The numerical gradient then becomes ill-defined
  (even in voxels that do not contribute to these bins).

*/
class PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBinTests
    : public ObjectiveFunctionTests<
          PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<DiscretisedDensity<3, float>>,
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
  PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBinTests(char const* const lm_data_filename,
                                                                                    char const* const density_filename = 0);
  void construct_input_data(shared_ptr<target_type>& density_sptr);

  void run_tests() override;

protected:
  char const* lm_data_filename;
  char const* density_filename;
  shared_ptr<CListModeData> lm_data_sptr;
  shared_ptr<ProjData> mult_proj_data_sptr;
  shared_ptr<ProjData> add_proj_data_sptr;
  shared_ptr<PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<target_type>> objective_function_sptr;

  //! run the test
  void run_tests_for_objective_function(objective_function_type& objective_function, target_type& target);
};

PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBinTests::
    PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBinTests(char const* lm_data_filename,
                                                                                      char const* const density_filename)
    : lm_data_filename(lm_data_filename),
      density_filename(density_filename)
{}

void
PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBinTests::run_tests_for_objective_function(
    objective_function_type& objective_function, target_type& target)
{
  // TODO enable this, but far too slow ATM
  // std::cerr << "----- testing Gradient\n";
  // test_gradient("PoissonLLProjData", objective_function, target, 0.01F);

  std::cerr << "----- testing concavity via Hessian-vector product (accumulate_Hessian_times_input)\n";
  test_Hessian_concavity("PoissonLLProjData", objective_function, target);

  std::cerr << "----- testing Hessian-vector product (accumulate_Hessian_times_input)\n";
  test_Hessian("PoissonLLProjData", objective_function, target, 0.5F);
}

void
PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBinTests::construct_input_data(
    shared_ptr<target_type>& density_sptr)
{
  lm_data_sptr = read_from_file<CListModeData>(this->lm_data_filename);

  if (this->density_filename == 0)
    {
      // construct a small image

      CartesianCoordinate3D<float> origin(0, 0, 0);
      const float zoom = 0.03F;

      density_sptr.reset(new VoxelsOnCartesianGrid<float>(
          lm_data_sptr->get_exam_info_sptr(), *lm_data_sptr->get_proj_data_info_sptr(), zoom, origin));
      write_to_file("target.hv", *density_sptr);
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

  auto proj_data_info_sptr = lm_data_sptr->get_proj_data_info_sptr()->create_shared_clone();
  // multiplicative term
  shared_ptr<BinNormalisation> bin_norm_sptr(new TrivialBinNormalisation());
  {

    mult_proj_data_sptr.reset(new ProjDataInMemory(lm_data_sptr->get_exam_info_sptr(), proj_data_info_sptr));
    for (int seg_num = proj_data_info_sptr->get_min_segment_num(); seg_num <= proj_data_info_sptr->get_max_segment_num();
         ++seg_num)
      {
        for (int timing_pos_num = proj_data_info_sptr->get_min_tof_pos_num();
             timing_pos_num <= proj_data_info_sptr->get_max_tof_pos_num();
             ++timing_pos_num)
          {
            SegmentByView<float> segment = proj_data_info_sptr->get_empty_segment_by_view(seg_num, false, timing_pos_num);
            // fill in some crazy values
            float value = 0;
            for (SegmentByView<float>::full_iterator iter = segment.begin_all(); iter != segment.end_all(); ++iter)
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
  add_proj_data_sptr.reset(new ProjDataInMemory(lm_data_sptr->get_exam_info_sptr(), proj_data_info_sptr));
  {
    for (int seg_num = proj_data_info_sptr->get_min_segment_num(); seg_num <= proj_data_info_sptr->get_max_segment_num();
         ++seg_num)
      {
        for (int timing_pos_num = proj_data_info_sptr->get_min_tof_pos_num();
             timing_pos_num <= proj_data_info_sptr->get_max_tof_pos_num();
             ++timing_pos_num)
          {
            SegmentByView<float> segment = proj_data_info_sptr->get_empty_segment_by_view(seg_num, false, timing_pos_num);
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

  objective_function_sptr.reset(new PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<target_type>);
  PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<target_type>& objective_function
      = reinterpret_cast<PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<target_type>&>(
          *objective_function_sptr);
  objective_function.set_input_data(lm_data_sptr);
  objective_function.set_use_subset_sensitivities(true);
  objective_function.set_max_segment_num_to_process(1);
  shared_ptr<ProjMatrixByBin> proj_matrix_sptr(new ProjMatrixByBinUsingRayTracing());
  objective_function.set_proj_matrix(proj_matrix_sptr);
  objective_function.set_normalisation_sptr(bin_norm_sptr);
  objective_function.set_additive_proj_data_sptr(add_proj_data_sptr);
  objective_function.set_num_subsets(2);
  if (!check(objective_function.set_up(density_sptr) == Succeeded::yes, "set-up of objective function"))
    return;
}

void
PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBinTests::run_tests()
{
  std::cerr << "Tests for PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin\n";
  const int verbosity_default = Verbosity::get();
  Verbosity::set(2);

#if 1
  shared_ptr<target_type> density_sptr;
  construct_input_data(density_sptr);
  this->run_tests_for_objective_function(*this->objective_function_sptr, *density_sptr);
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

int
main(int argc, char** argv)
{
  if (argc <= 1)
    error("Need to specify a list-mode filename");

  set_default_num_threads();

  PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBinTests tests(argv[1], argc > 2 ? argv[2] : 0);
  tests.run_tests();
  return tests.main_return_value();
}
