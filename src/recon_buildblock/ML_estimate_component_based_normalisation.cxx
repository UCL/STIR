/*
    Copyright (C) 2001- 2008, Hammersmith Imanet Ltd
    Copyright (C) 2019-2020, 2022, University College London
    Copyright (C) 2016-2017, PETsys Electronics
    Copyright (C) 2021, Gefei Chen
    Copyright (C) 2024, Robert Twyman Skelly
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
 \author Robert Twyman Skelly
 */
#include "stir/recon_buildblock/ML_estimate_component_based_normalisation.h"
#include "stir/ML_norm.h"
#include "stir/Scanner.h"
#include "stir/display.h"
#include "stir/info.h"
#include "stir/ProjData.h"
#include "stir/stream.h"
#include <string>
#include <utility>

START_NAMESPACE_STIR

void
ML_estimate_component_based_normalisation(const std::string& out_filename_prefix,
                                          const ProjData& measured_projdata,
                                          const ProjData& model_projdata,
                                          const int num_eff_iterations,
                                          const int num_iterations,
                                          const bool do_geo,
                                          const bool do_block,
                                          const bool do_symmetry_per_block,
                                          const bool do_KL,
                                          const bool do_display,
                                          const bool do_save_to_file)
{
  MLEstimateComponentBasedNormalisation estimator(out_filename_prefix,
                                                  measured_projdata,
                                                  model_projdata,
                                                  num_eff_iterations,
                                                  num_iterations,
                                                  do_geo,
                                                  do_block,
                                                  do_symmetry_per_block,
                                                  do_KL,
                                                  do_display,
                                                  do_save_to_file);
  estimator.process();
}

MLEstimateComponentBasedNormalisation::MLEstimateComponentBasedNormalisation(
    std::string out_filename_prefix_v,
    const ProjData& measured_projdata_v,
    const ProjData& model_projdata_v,
    const int num_eff_iterations_v,
    const int num_iterations_v,
    const bool do_geo_v,
    const bool do_block_v,
    const bool do_symmetry_per_block_v,
    const bool do_KL_v,
    const bool do_display_v,
    const bool do_save_to_file_v)
    : out_filename_prefix(std::move(out_filename_prefix_v)),
      num_eff_iterations(num_eff_iterations_v),
      num_iterations(num_iterations_v),
      do_geo(do_geo_v),
      do_block(do_block_v),
      do_symmetry_per_block(do_symmetry_per_block_v),
      do_KL(do_KL_v),
      do_display(do_display_v),
      do_save_to_file(do_save_to_file_v)
{
  const int num_transaxial_blocks
      = measured_projdata_v.get_proj_data_info_sptr()->get_scanner_sptr()->get_num_transaxial_blocks();
  const int num_axial_blocks = measured_projdata_v.get_proj_data_info_sptr()->get_scanner_sptr()->get_num_axial_blocks();
  const int virtual_axial_crystals
      = measured_projdata_v.get_proj_data_info_sptr()->get_scanner_sptr()->get_num_virtual_axial_crystals_per_block();
  const int virtual_transaxial_crystals
      = measured_projdata_v.get_proj_data_info_sptr()->get_scanner_sptr()->get_num_virtual_transaxial_crystals_per_block();
  const int num_physical_rings = measured_projdata_v.get_proj_data_info_sptr()->get_scanner_sptr()->get_num_rings()
                                 - (num_axial_blocks - 1) * virtual_axial_crystals;
  const int num_physical_detectors_per_ring
      = measured_projdata_v.get_proj_data_info_sptr()->get_scanner_sptr()->get_num_detectors_per_ring()
        - num_transaxial_blocks * virtual_transaxial_crystals;
  const int num_transaxial_buckets
      = measured_projdata_v.get_proj_data_info_sptr()->get_scanner_sptr()->get_num_transaxial_buckets();
  const int num_axial_buckets = measured_projdata_v.get_proj_data_info_sptr()->get_scanner_sptr()->get_num_axial_buckets();
  const int num_transaxial_blocks_per_bucket
      = measured_projdata_v.get_proj_data_info_sptr()->get_scanner_sptr()->get_num_transaxial_blocks_per_bucket();
  const int num_axial_blocks_per_bucket
      = measured_projdata_v.get_proj_data_info_sptr()->get_scanner_sptr()->get_num_axial_blocks_per_bucket();

  int num_physical_transaxial_crystals_per_basic_unit
      = measured_projdata_v.get_proj_data_info_sptr()->get_scanner_sptr()->get_num_transaxial_crystals_per_block()
        - virtual_transaxial_crystals;
  int num_physical_axial_crystals_per_basic_unit
      = measured_projdata_v.get_proj_data_info_sptr()->get_scanner_sptr()->get_num_axial_crystals_per_block()
        - virtual_axial_crystals;

  // If there are multiple buckets, we increase the symmetry size to a bucket. Otherwise, we use a block.
  if (do_symmetry_per_block == false)
    {
      num_physical_transaxial_crystals_per_basic_unit *= num_transaxial_blocks_per_bucket;
      num_physical_axial_crystals_per_basic_unit *= num_axial_blocks_per_bucket;
    }

  // Setup the data structures given the PET scanner geometry
  data_fan_sums = DetectorEfficiencies(IndexRange2D(num_physical_rings, num_physical_detectors_per_ring));
  efficiencies_ptr = std::make_shared<DetectorEfficiencies>(IndexRange2D(num_physical_rings, num_physical_detectors_per_ring));

  measured_geo_data = GeoData3D(num_physical_axial_crystals_per_basic_unit,
                                num_physical_transaxial_crystals_per_basic_unit / 2,
                                num_physical_rings,
                                num_physical_detectors_per_ring); // inputes have to be modified
  norm_geo_data_ptr = std::make_shared<GeoData3D>(num_physical_axial_crystals_per_basic_unit,
                                                  num_physical_transaxial_crystals_per_basic_unit / 2,
                                                  num_physical_rings,
                                                  num_physical_detectors_per_ring); // inputes have to be modified

  measured_block_data = BlockData3D(num_axial_blocks, num_transaxial_blocks, num_axial_blocks - 1, num_transaxial_blocks - 1);
  norm_block_data_ptr
      = std::make_shared<BlockData3D>(num_axial_blocks, num_transaxial_blocks, num_axial_blocks - 1, num_transaxial_blocks - 1);

  make_fan_data_remove_gaps(model_fan_data, model_projdata_v);
  make_fan_data_remove_gaps(measured_fan_data, measured_projdata_v);

  threshold_for_KL = compute_threshold_for_KL();

  make_fan_sum_data(data_fan_sums, measured_fan_data);
  make_geo_data(measured_geo_data, measured_fan_data);
  make_block_data(measured_block_data, measured_fan_data);
  if (do_display)
    {
      display(measured_block_data, "raw block data from measurements");
    }

  // Compute the do_KL specific varaibles from the measured data
  fan_sums = DetectorEfficiencies(IndexRange2D(num_physical_rings, num_physical_detectors_per_ring));
  geo_data = GeoData3D(num_physical_axial_crystals_per_basic_unit,
                       num_physical_transaxial_crystals_per_basic_unit / 2,
                       num_physical_rings,
                       num_physical_detectors_per_ring); // inputes have to be modified
  block_data = BlockData3D(num_axial_blocks, num_transaxial_blocks, num_axial_blocks - 1, num_transaxial_blocks - 1);
}

void
MLEstimateComponentBasedNormalisation::process()
{
  if (do_display)
    {
      display(model_fan_data, "model");
    }

  // Initialize the efficiencies, geo data and block data to 1
  efficiencies_ptr->fill(sqrt(data_fan_sums.sum() / model_fan_data.sum()));
  norm_geo_data_ptr->fill(1);
  norm_block_data_ptr->fill(1);

  for (int iter_num = 1; iter_num <= std::max(num_iterations, 1); ++iter_num)
    {
      fan_data = model_fan_data;
      apply_geo_norm(fan_data, *norm_geo_data_ptr);
      apply_block_norm(fan_data, *norm_block_data_ptr);
      if (do_display)
        {
          display(fan_data, "model*geo*block");
        }

      // Internal Efficiency iterations loop
      for (int eff_iter_num = 1; eff_iter_num <= num_eff_iterations; ++eff_iter_num)
        {
          efficiency_iteration(iter_num, eff_iter_num);
        }

      if (do_geo)
        {
          // geo norm iteration
          geo_normalization_iteration(iter_num); // Calculate geo norm iteration
        }
      if (do_block)
        {
          // block norm iteration
          block_normalization_iteration(iter_num); // Calculate block norm iteration
        }
      //// print KL for fansums
      if (do_KL)
        {
          make_fan_sum_data(fan_sums, fan_data);
          make_geo_data(geo_data, fan_data);
          make_block_data(block_data, measured_fan_data);
          info(boost::format("KL on fans: %1%, %2%") % KL(measured_fan_data, fan_data, 0) % KL(measured_geo_data, geo_data, 0));
        }
    }
  data_processed = true;
}
bool
MLEstimateComponentBasedNormalisation::has_processed_data() const
{
  return data_processed;
}

std::shared_ptr<DetectorEfficiencies>
MLEstimateComponentBasedNormalisation::get_efficiencies() const
{
  if (!data_processed)
    {
      return nullptr;
    }
  return efficiencies_ptr;
}

std::shared_ptr<GeoData3D>
MLEstimateComponentBasedNormalisation::get_geo_data() const
{
  if (!data_processed)
    {
      return nullptr;
    }
  return norm_geo_data_ptr;
}

std::shared_ptr<BlockData3D>
MLEstimateComponentBasedNormalisation::get_block_data() const
{
  if (!data_processed)
    {
      return nullptr;
    }
  return norm_block_data_ptr;
}

void
MLEstimateComponentBasedNormalisation::efficiency_iteration(const int iter_num, const int eff_iter_num)
{
  iterate_efficiencies(*efficiencies_ptr, data_fan_sums, fan_data);
  if (do_save_to_file)
    {
      write_efficiencies_to_file(iter_num, eff_iter_num);
    }
  if (do_KL)
    {
      apply_efficiencies(fan_data, *efficiencies_ptr);
      std::cerr << "measured*norm min " << measured_fan_data.find_min() << " ,max " << measured_fan_data.find_max() << std::endl;
      std::cerr << "model*norm min " << fan_data.find_min() << " ,max " << fan_data.find_max() << std::endl;
      if (do_display)
        display(fan_data, "model_times_norm");
      info(boost::format("KL %1%") % KL(measured_fan_data, fan_data, threshold_for_KL));
      // now restore for further iterations
      fan_data = model_fan_data;
      apply_geo_norm(fan_data, *norm_geo_data_ptr);
      apply_block_norm(fan_data, *norm_block_data_ptr);
    }
  if (do_display)
    {
      fan_data.fill(1);
      apply_efficiencies(fan_data, *efficiencies_ptr);
      display(fan_data, "eff norm");
      // now restore for further iterations
      fan_data = model_fan_data;
      apply_geo_norm(fan_data, *norm_geo_data_ptr);
      apply_block_norm(fan_data, *norm_block_data_ptr);
    }
}

void
MLEstimateComponentBasedNormalisation::geo_normalization_iteration(int iter_num)
{

  fan_data = model_fan_data;                        // Reset fan_data to model_data
  apply_efficiencies(fan_data, *efficiencies_ptr);  // Apply efficiencies
  apply_block_norm(fan_data, *norm_block_data_ptr); // Apply block norm
  iterate_geo_norm(*norm_geo_data_ptr, measured_geo_data, fan_data);

  if (do_save_to_file)
    {
      write_geo_data_to_file(iter_num);
    }
  if (do_KL)
    {
      apply_geo_norm(fan_data, *norm_geo_data_ptr);
      info(boost::format("KL %1%") % KL(measured_fan_data, fan_data, threshold_for_KL));
    }
  if (do_display)
    {
      fan_data.fill(1);
      apply_geo_norm(fan_data, *norm_geo_data_ptr);
      display(fan_data, "geo norm");
    }
}

void
MLEstimateComponentBasedNormalisation::block_normalization_iteration(const int iter_num)
{

  fan_data = model_fan_data;                                               // Reset fan_data to model_data
  apply_efficiencies(fan_data, *efficiencies_ptr);                         // Apply efficiencies
  apply_geo_norm(fan_data, *norm_geo_data_ptr);                            // Apply geo norm
  iterate_block_norm(*norm_block_data_ptr, measured_block_data, fan_data); // Iterate block norm calculation

  if (do_save_to_file)
    {
      write_block_data_to_file(iter_num);
    }
  if (do_KL)
    {
      apply_block_norm(fan_data, *norm_block_data_ptr);
      info(boost::format("KL %1%") % KL(measured_fan_data, fan_data, threshold_for_KL));
    }
  if (do_display)
    {
      fan_data.fill(1);
      apply_block_norm(fan_data, *norm_block_data_ptr);
      display(*norm_block_data_ptr, "raw block norm");
      display(fan_data, "block norm");
    }
}

void
MLEstimateComponentBasedNormalisation::write_efficiencies_to_file(const int iter_num, const int eff_iter_num) const
{
  char* out_filename = new char[out_filename_prefix.size() + 30];
  sprintf(out_filename, "%s_%s_%d_%d.out", out_filename_prefix.c_str(), "eff", iter_num, eff_iter_num);
  std::ofstream out(out_filename);
  out << *efficiencies_ptr;
}

void
MLEstimateComponentBasedNormalisation::write_geo_data_to_file(const int iter_num) const
{
  char* out_filename = new char[out_filename_prefix.size() + 30];
  sprintf(out_filename, "%s_%s_%d.out", out_filename_prefix.c_str(), "geo", iter_num);
  std::ofstream out(out_filename);
  out << *norm_geo_data_ptr;
}

void
MLEstimateComponentBasedNormalisation::write_block_data_to_file(const int iter_num) const
{
  char* out_filename = new char[out_filename_prefix.size() + 30];
  sprintf(out_filename, "%s_%s_%d.out", out_filename_prefix.c_str(), "block", iter_num);
  std::ofstream out(out_filename);
  out << *norm_block_data_ptr;
}

float
MLEstimateComponentBasedNormalisation::compute_threshold_for_KL()
{
  // Set the max found value to -inf
  float max_elem = -std::numeric_limits<float>::infinity();

  const auto min_ra = model_fan_data.get_min_ra();
  const auto max_ra = model_fan_data.get_max_ra();
  const auto min_a = model_fan_data.get_min_a();
  const auto max_a = model_fan_data.get_max_a();

  for (auto ra = min_ra; ra <= max_ra; ++ra)
    {
      const auto min_rb = std::max(ra, model_fan_data.get_min_rb(ra));
      const auto max_rb = model_fan_data.get_max_rb(ra);

      for (auto a = min_a; a <= max_a; ++a)
        {
          const auto min_b = model_fan_data.get_min_b(a);
          const auto max_b = model_fan_data.get_max_b(a);

          for (auto rb = min_rb; rb <= max_rb; ++rb)
            {
              for (auto b = min_b; b <= max_b; ++b)
                {
                  if (model_fan_data(ra, a, rb, b) == 0)
                    {
                      max_elem = std::max(max_elem, measured_fan_data(ra, a, rb, b));
                    }
                }
            }
        }
    }

  return max_elem / 100000.F;
}

END_NAMESPACE_STIR
