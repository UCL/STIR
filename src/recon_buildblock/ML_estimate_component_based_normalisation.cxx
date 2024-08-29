/*
    Copyright (C) 2001- 2008, Hammersmith Imanet Ltd
    Copyright (C) 2019-2020, 2022, University College London
    Copyright (C) 2016-2017, PETsys Electronics
    Copyright (C) 2021, Gefei Chen
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!

 \file
 \ingroup recon_buildblock
 \brief Implementation of ML_estimate_component_based_normalisation

 \author Kris Thielemans
 \author Tahereh Niknejad
 \author Gefei Chen
 */
#include "stir/recon_buildblock/ML_estimate_component_based_normalisation.h"

#include "stir/ML_norm.h"
#include "stir/Scanner.h"
#include "stir/stream.h"
#include "stir/display.h"
#include "stir/info.h"
#include "stir/warning.h"
#include "stir/ProjData.h"
#include <boost/format.hpp>
#include <fstream>
#include <string>
#include <algorithm>

START_NAMESPACE_STIR

void
ML_estimate_component_based_normalisation(const std::string& out_filename_prefix,
                                          const ProjData& measured_data,
                                          const ProjData& model_data,
                                          int num_eff_iterations,
                                          int num_iterations,
                                          bool do_geo,
                                          bool do_block,
                                          bool do_symmetry_per_block,
                                          bool do_KL,
                                          bool do_display)
{
  MLEstimateComponentBasedNormalisation estimator(out_filename_prefix,
                                                  measured_data,
                                                  model_data,
                                                  num_eff_iterations,
                                                  num_iterations,
                                                  do_geo,
                                                  do_block,
                                                  do_symmetry_per_block,
                                                  do_KL,
                                                  do_display);
  estimator.run();
}

MLEstimateComponentBasedNormalisation::MLEstimateComponentBasedNormalisation(const std::string& out_filename_prefix,
                                                                             const ProjData& measured_data,
                                                                             const ProjData& model_data,
                                                                             int num_eff_iterations,
                                                                             int num_iterations,
                                                                             bool do_geo,
                                                                             bool do_block,
                                                                             bool do_symmetry_per_block,
                                                                             bool do_KL,
                                                                             bool do_display)
    : measured_data(measured_data),
      model_data(model_data),
      out_filename_prefix(out_filename_prefix),
      num_eff_iterations(num_eff_iterations),
      num_iterations(num_iterations),
      do_geo(do_geo),
      do_block(do_block),
      do_symmetry_per_block(do_symmetry_per_block),
      do_KL(do_KL),
      do_display(do_display)
{}

//! Helper function to write efficiencies to a file
void
MLEstimateComponentBasedNormalisation::write_efficiencies_to_file(int iter_num,
                                                                  int eff_iter_num,
                                                                  const DetectorEfficiencies& efficiencies)
{
  char* out_filename = new char[out_filename_prefix.size() + 30];
  sprintf(out_filename, "%s_%s_%d_%d.out", out_filename_prefix.c_str(), "eff", iter_num, eff_iter_num);
  std::ofstream out(out_filename);
  out << efficiencies;
}

//! Helper function to write geo data to a file
void
MLEstimateComponentBasedNormalisation::write_geo_data_to_file(int iter_num, const GeoData3D& norm_geo_data)
{
  char* out_filename = new char[out_filename_prefix.size() + 30];
  sprintf(out_filename, "%s_%s_%d.out", out_filename_prefix.c_str(), "geo", iter_num);
  std::ofstream out(out_filename);
  out << norm_geo_data;
}

//! Helper function to write block data to a file
void
MLEstimateComponentBasedNormalisation::write_block_data_to_file(int iter_num, const BlockData3D& norm_block_data)
{
  char* out_filename = new char[out_filename_prefix.size() + 30];
  sprintf(out_filename, "%s_%s_%d.out", out_filename_prefix.c_str(), "block", iter_num);
  std::ofstream out(out_filename);
  out << norm_block_data;
}

// Function to compute factors dependent on the data
void
MLEstimateComponentBasedNormalisation::compute_initial_data_dependent_factors(const FanProjData& model_fan_data,
                                                                              FanProjData& measured_fan_data,
                                                                              DetectorEfficiencies& data_fan_sums,
                                                                              GeoData3D& measured_geo_data,
                                                                              BlockData3D& measured_block_data)
{
  make_fan_data_remove_gaps(measured_fan_data, measured_data);

  /* TEMP FIX */
  for (int ra = model_fan_data.get_min_ra(); ra <= model_fan_data.get_max_ra(); ++ra)
    {
      for (int a = model_fan_data.get_min_a(); a <= model_fan_data.get_max_a(); ++a)
        {
          for (int rb = std::max(ra, model_fan_data.get_min_rb(ra)); rb <= model_fan_data.get_max_rb(ra); ++rb)
            {
              for (int b = model_fan_data.get_min_b(a); b <= model_fan_data.get_max_b(a); ++b)
                if (model_fan_data(ra, a, rb, b) == 0)
                  measured_fan_data(ra, a, rb, b) = 0;
            }
        }
    }

  threshold_for_KL = measured_fan_data.find_max() / 100000.F;
  // display(measured_fan_data, "measured data");

  make_fan_sum_data(data_fan_sums, measured_fan_data);
  make_geo_data(measured_geo_data, measured_fan_data);
  make_block_data(measured_block_data, measured_fan_data);
  if (do_display)
    display(measured_block_data, "raw block data from measurements");
}

// Function to handle efficiency iteration
void
MLEstimateComponentBasedNormalisation::efficiency_iteration(FanProjData& fan_data,
                                                            const FanProjData& model_fan_data,
                                                            const GeoData3D& norm_geo_data,
                                                            const BlockData3D& norm_block_data,
                                                            DetectorEfficiencies& efficiencies,
                                                            const DetectorEfficiencies& data_fan_sums,
                                                            int iter_num,
                                                            int eff_iter_num,
                                                            const FanProjData& measured_fan_data)
{
  iterate_efficiencies(efficiencies, data_fan_sums, fan_data);
  write_efficiencies_to_file(iter_num, eff_iter_num, efficiencies);
  if (do_KL)
    {
      apply_efficiencies(fan_data, efficiencies);
      std::cerr << "measured*norm min " << measured_fan_data.find_min() << " ,max " << measured_fan_data.find_max() << std::endl;
      std::cerr << "model*norm min " << fan_data.find_min() << " ,max " << fan_data.find_max() << std::endl;
      if (do_display)
        display(fan_data, "model_times_norm");
      info(boost::format("KL %1%") % KL(measured_fan_data, fan_data, threshold_for_KL));
      // now restore for further iterations
      fan_data = model_fan_data;
      apply_geo_norm(fan_data, norm_geo_data);
      apply_block_norm(fan_data, norm_block_data);
    }
  if (do_display)
    {
      fan_data.fill(1);
      apply_efficiencies(fan_data, efficiencies);
      display(fan_data, "eff norm");
      // now restore for further iterations
      fan_data = model_fan_data;
      apply_geo_norm(fan_data, norm_geo_data);
      apply_block_norm(fan_data, norm_block_data);
    }
}

// Function to handle geo normalization
void
MLEstimateComponentBasedNormalisation::geo_normalization_iteration(FanProjData& fan_data,
                                                                   const FanProjData& model_fan_data,
                                                                   const DetectorEfficiencies& efficiencies,
                                                                   const BlockData3D& norm_block_data,
                                                                   GeoData3D& norm_geo_data,
                                                                   const GeoData3D& measured_geo_data,
                                                                   int iter_num,
                                                                   const FanProjData& measured_fan_data)
{
  fan_data = model_fan_data;
  apply_efficiencies(fan_data, efficiencies);
  apply_block_norm(fan_data, norm_block_data);

  if (do_geo)
    {
      iterate_geo_norm(norm_geo_data, measured_geo_data, fan_data);
    }

  write_geo_data_to_file(iter_num, norm_geo_data);
  if (do_KL)
    {
      apply_geo_norm(fan_data, norm_geo_data);
      info(boost::format("KL %1%") % KL(measured_fan_data, fan_data, threshold_for_KL));
    }
  if (do_display)
    {
      fan_data.fill(1);
      apply_geo_norm(fan_data, norm_geo_data);
      display(fan_data, "geo norm");
    }
}

// Function to handle block normalization
void
MLEstimateComponentBasedNormalisation::block_normalization_iteration(FanProjData& fan_data,
                                                                     const FanProjData& model_fan_data,
                                                                     const DetectorEfficiencies& efficiencies,
                                                                     const GeoData3D& norm_geo_data,
                                                                     BlockData3D& norm_block_data,
                                                                     const BlockData3D& measured_block_data,
                                                                     int iter_num,
                                                                     const FanProjData& measured_fan_data)
{
  fan_data = model_fan_data;
  apply_efficiencies(fan_data, efficiencies);
  apply_geo_norm(fan_data, norm_geo_data);
  if (do_block)
    {
      iterate_block_norm(norm_block_data, measured_block_data, fan_data);
    }
  write_block_data_to_file(iter_num, norm_block_data);
  if (do_KL)
    {
      apply_block_norm(fan_data, norm_block_data);
      info(boost::format("KL %1%") % KL(measured_fan_data, fan_data, threshold_for_KL));
    }
  if (do_display)
    {
      fan_data.fill(1);
      apply_block_norm(fan_data, norm_block_data);
      display(norm_block_data, "raw block norm");
      display(fan_data, "block norm");
    }
}

void
MLEstimateComponentBasedNormalisation::run()
{
  const int num_transaxial_blocks = measured_data.get_proj_data_info_sptr()->get_scanner_sptr()->get_num_transaxial_blocks();
  const int num_axial_blocks = measured_data.get_proj_data_info_sptr()->get_scanner_sptr()->get_num_axial_blocks();
  const int virtual_axial_crystals
      = measured_data.get_proj_data_info_sptr()->get_scanner_sptr()->get_num_virtual_axial_crystals_per_block();
  const int virtual_transaxial_crystals
      = measured_data.get_proj_data_info_sptr()->get_scanner_sptr()->get_num_virtual_transaxial_crystals_per_block();
  const int num_physical_rings = measured_data.get_proj_data_info_sptr()->get_scanner_sptr()->get_num_rings()
                                 - (num_axial_blocks - 1) * virtual_axial_crystals;
  const int num_physical_detectors_per_ring
      = measured_data.get_proj_data_info_sptr()->get_scanner_sptr()->get_num_detectors_per_ring()
        - num_transaxial_blocks * virtual_transaxial_crystals;
  const int num_transaxial_buckets = measured_data.get_proj_data_info_sptr()->get_scanner_sptr()->get_num_transaxial_buckets();
  const int num_axial_buckets = measured_data.get_proj_data_info_sptr()->get_scanner_sptr()->get_num_axial_buckets();
  const int num_transaxial_blocks_per_bucket
      = measured_data.get_proj_data_info_sptr()->get_scanner_sptr()->get_num_transaxial_blocks_per_bucket();
  const int num_axial_blocks_per_bucket
      = measured_data.get_proj_data_info_sptr()->get_scanner_sptr()->get_num_axial_blocks_per_bucket();

  int num_physical_transaxial_crystals_per_basic_unit
      = measured_data.get_proj_data_info_sptr()->get_scanner_sptr()->get_num_transaxial_crystals_per_block()
        - virtual_transaxial_crystals;
  int num_physical_axial_crystals_per_basic_unit
      = measured_data.get_proj_data_info_sptr()->get_scanner_sptr()->get_num_axial_crystals_per_block() - virtual_axial_crystals;
  // If there are multiple buckets, we increase the symmetry size to a bucket. Otherwise, we use a block.
  if (do_symmetry_per_block == false)
    {
      if (num_transaxial_buckets > 1)
        {
          num_physical_transaxial_crystals_per_basic_unit *= num_transaxial_blocks_per_bucket;
        }
      if (num_axial_buckets > 1)
        {
          num_physical_axial_crystals_per_basic_unit *= num_axial_blocks_per_bucket;
        }
    }

  FanProjData model_fan_data;
  FanProjData fan_data;
  DetectorEfficiencies data_fan_sums(IndexRange2D(num_physical_rings, num_physical_detectors_per_ring));
  DetectorEfficiencies efficiencies(IndexRange2D(num_physical_rings, num_physical_detectors_per_ring));

  GeoData3D measured_geo_data(num_physical_axial_crystals_per_basic_unit,
                              num_physical_transaxial_crystals_per_basic_unit / 2,
                              num_physical_rings,
                              num_physical_detectors_per_ring); // inputes have to be modified
  GeoData3D norm_geo_data(num_physical_axial_crystals_per_basic_unit,
                          num_physical_transaxial_crystals_per_basic_unit / 2,
                          num_physical_rings,
                          num_physical_detectors_per_ring); // inputes have to be modified

  BlockData3D measured_block_data(num_axial_blocks, num_transaxial_blocks, num_axial_blocks - 1, num_transaxial_blocks - 1);
  BlockData3D norm_block_data(num_axial_blocks, num_transaxial_blocks, num_axial_blocks - 1, num_transaxial_blocks - 1);

  make_fan_data_remove_gaps(model_fan_data, model_data);

  // next could be local if KL is not computed below
  FanProjData measured_fan_data;

  compute_initial_data_dependent_factors(
      model_fan_data, measured_fan_data, data_fan_sums, measured_geo_data, measured_block_data);

  // std::cerr << "model min " << model_fan_data.find_min() << " ,max " << model_fan_data.find_max() << std::endl;
  if (do_display)
    display(model_fan_data, "model");

  for (int iter_num = 1; iter_num <= std::max(num_iterations, 1); ++iter_num)
    {
      if (iter_num == 1)
        {
          efficiencies.fill(sqrt(data_fan_sums.sum() / model_fan_data.sum()));
          norm_geo_data.fill(1);
          norm_block_data.fill(1);
        }
      fan_data = model_fan_data;
      apply_geo_norm(fan_data, norm_geo_data);
      apply_block_norm(fan_data, norm_block_data);
      if (do_display)
        {
          display(fan_data, "model*geo*block");
        }

      // Efficiency iterations
      for (int eff_iter_num = 1; eff_iter_num <= num_eff_iterations; ++eff_iter_num)
        {
          efficiency_iteration(fan_data,
                               model_fan_data,
                               norm_geo_data,
                               norm_block_data,
                               efficiencies,
                               data_fan_sums,
                               iter_num,
                               eff_iter_num,
                               measured_fan_data);
        }

      // geo norm
      geo_normalization_iteration(
          fan_data, model_fan_data, efficiencies, norm_block_data, norm_geo_data, measured_geo_data, iter_num, measured_fan_data);

      // block norm
      block_normalization_iteration(fan_data,
                                    model_fan_data,
                                    efficiencies,
                                    norm_geo_data,
                                    norm_block_data,
                                    measured_block_data,
                                    iter_num,
                                    measured_fan_data);

      //// print KL for fansums
      if (do_KL)
        {
          DetectorEfficiencies fan_sums(IndexRange2D(num_physical_rings, num_physical_detectors_per_ring));
          GeoData3D geo_data(num_physical_axial_crystals_per_basic_unit,
                             num_physical_transaxial_crystals_per_basic_unit / 2,
                             num_physical_rings,
                             num_physical_detectors_per_ring); // inputes have to be modified
          BlockData3D block_data(num_axial_blocks, num_transaxial_blocks, num_axial_blocks - 1, num_transaxial_blocks - 1);

          make_fan_sum_data(fan_sums, fan_data);
          make_geo_data(geo_data, fan_data);
          make_block_data(block_data, measured_fan_data);

          info(boost::format("KL on fans: %1%, %2") % KL(measured_fan_data, fan_data, 0) % KL(measured_geo_data, geo_data, 0));
        }
    }
}

END_NAMESPACE_STIR
