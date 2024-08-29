/*
    Copyright (C) 2022, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
 \file
 \ingroup recon_buildblock
 \brief Declaration of stir::ML_estimate_component_based_normalisation
 \author Kris Thielemans
 */
#include "stir/common.h"
#include <string>
#include "stir/ProjData.h"
#include "stir/ML_norm.h"

START_NAMESPACE_STIR

/*!
 \brief Find normalisation factors using a maximum likelihood approach

  \ingroup recon_buildblock
*/

// The function that calls the class MLEstimateComponentBasedNormalisation and runs the normalisation estimation
void ML_estimate_component_based_normalisation(const std::string& out_filename_prefix,
                                               const ProjData& measured_data,
                                               const ProjData& model_data,
                                               int num_eff_iterations,
                                               int num_iterations,
                                               bool do_geo,
                                               bool do_block,
                                               bool do_symmetry_per_block,
                                               bool do_KL,
                                               bool do_display);

class MLEstimateComponentBasedNormalisation
{
public:
  MLEstimateComponentBasedNormalisation(const std::string& out_filename_prefix,
                                        const ProjData& measured_data,
                                        const ProjData& model_data,
                                        int num_eff_iterations,
                                        int num_iterations,
                                        bool do_geo,
                                        bool do_block,
                                        bool do_symmetry_per_block,
                                        bool do_KL,
                                        bool do_display);

  void run();

private:
  void write_efficiencies_to_file(int iter_num, int eff_iter_num, const DetectorEfficiencies& efficiencies);

  void write_geo_data_to_file(int iter_num, const GeoData3D& norm_geo_data);

  void write_block_data_to_file(int iter_num, const BlockData3D& norm_block_data);

  void compute_initial_data_dependent_factors(const FanProjData& model_fan_data,
                                              FanProjData& measured_fan_data,
                                              DetectorEfficiencies& data_fan_sums,
                                              GeoData3D& measured_geo_data,
                                              BlockData3D& measured_block_data);

  void efficiency_iteration(FanProjData& fan_data,
                            const FanProjData& model_fan_data,
                            const GeoData3D& norm_geo_data,
                            const BlockData3D& norm_block_data,
                            DetectorEfficiencies& efficiencies,
                            const DetectorEfficiencies& data_fan_sums,
                            int iter_num,
                            int eff_iter_num,
                            const FanProjData& measured_fan_data);

  void geo_normalization_iteration(FanProjData& fan_data,
                                   const FanProjData& model_fan_data,
                                   const DetectorEfficiencies& efficiencies,
                                   const BlockData3D& norm_block_data,
                                   GeoData3D& norm_geo_data,
                                   const GeoData3D& measured_geo_data,
                                   int iter_num,
                                   const FanProjData& measured_fan_data);

  void block_normalization_iteration(FanProjData& fan_data,
                                     const FanProjData& model_fan_data,
                                     const DetectorEfficiencies& efficiencies,
                                     const GeoData3D& norm_geo_data,
                                     BlockData3D& norm_block_data,
                                     const BlockData3D& measured_block_data,
                                     int iter_num,
                                     const FanProjData& measured_fan_data);
  // Arguments
  std::string out_filename_prefix;
  const ProjData& measured_data;
  const ProjData& model_data;
  int num_eff_iterations;
  int num_iterations;
  bool do_geo;
  bool do_block;
  bool do_symmetry_per_block;
  bool do_KL;
  bool do_display;

  // Calculated values
  float threshold_for_KL;
};

END_NAMESPACE_STIR
