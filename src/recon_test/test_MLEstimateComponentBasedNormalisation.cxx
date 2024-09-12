/*
Copyright (C) 2024, Robert Twyman skelly
    This file is part of STIR.
    SPDX-License-Identifier: Apache-2.0
    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup recon_test
  \brief Test program to ensure MLEstimateComponentBasedNormalisation works as expected and
         produces the correct normalization factors from components
  \author Robert Twyman Skelly
*/

#include "stir/RunTests.h"
#include "stir/num_threads.h"
#include "stir/recon_buildblock/ForwardProjectorByBinUsingProjMatrixByBin.h"
#include "stir/ProjDataInMemory.h"
#include "stir/SeparableCartesianMetzImageFilter.h"
#include "stir/recon_buildblock/MLEstimateComponentBasedNormalisation.h"

START_NAMESPACE_STIR

/*!
  \ingroup test
  \brief Test class for MLEstimateComponentBasedNormalisation
*/
class MLEstimateComponentBasedNormalisationTest : public RunTests
{
public:
  //! Constructor that can take some input data to run the test with
  MLEstimateComponentBasedNormalisationTest() = default;

  ~MLEstimateComponentBasedNormalisationTest() override = default;

  void run_tests() override
  {

    const auto scanner = std::make_shared<Scanner>(*Scanner::get_scanner_from_name("Discovery 690"));
    const auto exam_info = std::make_shared<ExamInfo>(ImagingModality::PT);
    exam_info->patient_position = PatientPosition(PatientPosition::HFS);

    const shared_ptr<ProjDataInfo> projdata_info(ProjDataInfo::ProjDataInfoCTI(
        /*scanner_ptr*/ scanner,
        /*span*/ 1,
        /*max_delta*/ 23,
        /*views*/ scanner->get_num_detectors_per_ring() / 2,
        /*tang_pos*/ scanner->get_default_num_arccorrected_bins(),
        /*arc_corrected*/ false,
        /*tof_mash_factor*/ -1));

    const auto measured_projdata = std::make_shared<ProjDataInMemory>(ProjDataInMemory(exam_info, projdata_info, false));
    measured_projdata->fill(1.F);

    const auto model_projdata = std::make_shared<ProjDataInMemory>(*measured_projdata);
    model_projdata->fill(2.F);

    constexpr int num_eff_iterations = 6;
    constexpr int num_iterations = 2;
    constexpr bool do_geo = false;
    constexpr bool do_block = false;
    constexpr bool do_symmetry_per_block = false;
    constexpr bool do_KL = false;
    constexpr bool do_display = false;
    constexpr bool do_save_to_file = false;
    auto ml_estimator = MLEstimateComponentBasedNormalisation("",
                                                              *measured_projdata,
                                                              *model_projdata,
                                                              num_eff_iterations,
                                                              num_iterations,
                                                              do_geo,
                                                              do_block,
                                                              do_symmetry_per_block,
                                                              do_KL,
                                                              do_display,
                                                              do_save_to_file);
    ml_estimator.process();

    // Check the efficiencies, with measured data as uniform 1s and model data as uniform 2s, expect this to be ~0.707
    auto efficiencies = ml_estimator.get_efficiencies();
    check_if_less(efficiencies.find_max(), 0.75, "The max value of the efficiencies is greater than expected");
    check_if_less(0.65, efficiencies.find_max(), "The max value of the efficiencies is less than expected");
    check_if_less(efficiencies.find_min(), 0.75, "The min value of the efficiencies is greater than expected");
    check_if_less(0.65, efficiencies.find_min(), "The min value of the efficiencies is less than expected");

    auto bin_normalization = ml_estimator.construct_bin_normfactors_from_components();
    bin_normalization.set_up(measured_projdata->get_exam_info_sptr(), measured_projdata->get_proj_data_info_sptr());

    // Compute the projdata
    ProjDataInMemory normalization_projdata(*measured_projdata);
    normalization_projdata.fill(1.F);

    bin_normalization.apply(normalization_projdata);
    check_if_less(
        normalization_projdata.find_max(), 2.f * 1.1f, "The max value of the normalization projdata is greater than expected");
    check_if_less(
        2.f * 0.9f, normalization_projdata.find_max(), "The max value of the normalization projdata is less than expected");
    check_if_less(
        normalization_projdata.find_min(), 2.f * 1.1f, "The min value of the normalization projdata is greater than expected");
    check_if_less(
        2.f * 0.9f, normalization_projdata.find_min(), "The min value of the normalization projdata is less than expected");

    // ASSERT_EQ()
  }
};
END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int
main()
{
  Verbosity::set(1);
  MLEstimateComponentBasedNormalisationTest tests;
  tests.run_tests();
  return tests.main_return_value();
}
