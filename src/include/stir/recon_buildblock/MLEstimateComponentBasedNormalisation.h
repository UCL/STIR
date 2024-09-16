/*
    Copyright (C) 2022, University College London
    Copyright (C) 2024, Robert Twyman Skelly
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
 \file
 \ingroup recon_buildblock
 \brief Declaration of stir::ML_estimate_component_based_normalisation
 \author Kris Thielemans
 \author Robert Twyman Skelly
 */
#include "BinNormalisationPETFromComponents.h"
#include "stir/common.h"
#include <string>
#include "stir/ProjData.h"
#include "stir/ML_norm.h"
#include "stir/ProjDataInMemory.h"

START_NAMESPACE_STIR

/*!
  \brief Find normalisation factors using a maximum likelihood approach
  \ingroup recon_buildblock
        \param out_filename_prefix The prefix for the output files
        \param measured_data The measured projection data
        \param model_data The model projection data
        \param num_eff_iterations The number of (sub-)efficiency iterations to perform per iteration of the algorithm
        \param num_iterations The number of algorithm iterations to perform
        \param do_geo Whether to perform geo normalization calculations
        \param do_block Whether to perform block normalization calculations
        \param do_symmetry_per_block Whether to perform block normalization calculations with symmetry
        \param do_KL Whether to perform Kullback-Leibler divergence calculations and display the KL value. This can be slow.
        \param do_display Whether to display the progress of the algorithm.
        \param do_save_to_file Whether to save the each iteration of the efficiencies, geo data and block data to file.
*/
void ML_estimate_component_based_normalisation(const std::string& out_filename_prefix,
                                               const ProjData& measured_projdata,
                                               const ProjData& model_projdata,
                                               int num_eff_iterations,
                                               int num_iterations,
                                               bool do_geo,
                                               bool do_block,
                                               bool do_symmetry_per_block,
                                               bool do_KL,
                                               bool do_display,
                                               bool do_save_to_file = true);

/*!
  \brief Find normalisation factors using a maximum likelihood approach
  \ingroup recon_buildblock
*/
class MLEstimateComponentBasedNormalisation
{
public:
  /*!
   \brief Constructor
    \param out_filename_prefix_v The prefix for the output files
    \param measured_projdata_v The measured projection data
    \param model_projdata_v The model projection data
    \param num_eff_iterations_v The number of (sub-)efficiency iterations to perform per iteration of the algorithm
    \param num_iterations_v The number of algorithm iterations to perform
    \param do_geo_v Whether to perform geo normalization calculations
    \param do_block_v Whether to perform block normalization calculations
    \param do_symmetry_per_block_v Whether to perform block normalization calculations with symmetry
    \param do_KL_v Whether to perform Kullback-Leibler divergence calculations and display the KL value. This can be slow.
    \param do_display_v Whether to display the progress of the algorithm.
    \param do_save_to_file_v Whether to save the each iteration of the efficiencies, geo data and block data to file.
  */
  MLEstimateComponentBasedNormalisation(std::string out_filename_prefix_v,
                                        const ProjData& measured_projdata_v,
                                        const ProjData& model_projdata_v,
                                        int num_eff_iterations_v,
                                        int num_iterations_v,
                                        bool do_geo_v,
                                        bool do_block_v,
                                        bool do_symmetry_per_block_v,
                                        bool do_KL_v,
                                        bool do_display_v,
                                        bool do_save_to_file_v);

  /*!
  \brief Run the normalisation estimation algorithm using the parameters provided in the constructor.
  */
  void process();

  //! Check if the data has been processed
  bool get_data_is_processed() const;

  //! Get the efficiencies, requires process() to be called first
  DetectorEfficiencies get_efficiencies() const;
  //! Get the geo data, requires process() to be called first and do_geo to be true
  GeoData3D get_geo_data() const;
  //! Get the block data, requires process() to be called first and do_block to be true
  BlockData3D get_block_data() const;

  //! Construct a BinNormalisationPETFromComponents from the calculated efficiencies, geo data and block data
  BinNormalisationPETFromComponents construct_bin_norm_from_pet_components() const;

  /*!
  \brief Get the output filename prefix.
  \return The output filename prefix.
  */
  std::string get_output_filename_prefix() const { return out_filename_prefix; }

  /*!
    \brief Set the output filename prefix.
    \param out_filename_prefix The new output filename prefix.
  */
  void set_output_filename_prefix(const std::string& out_filename_prefix);

  /*!
    \brief Get the number of efficiency iterations.
    There are the inner iterations of the algorithm, num_eff_iterations are performed per iteration.
    \return The number of efficiency iterations.
  */
  int get_num_eff_iterations() const { return num_eff_iterations; }

  /*!
    \brief Set the number of efficiency iterations.
    There are the inner iterations of the algorithm, num_eff_iterations are performed per iteration.
    \param num_eff_iterations The new number of efficiency iterations.
  */
  void set_num_eff_iterations(int num_eff_iterations);

  /*!
    \brief Get the number of iterations.
    These are the outer iterations of the algorithm.
    \return The number of iterations.
  */
  int get_num_iterations() const { return num_iterations; }

  /*!
    \brief Set the number of iterations.
    These are the outer iterations of the algorithm.
    \param num_iterations The new number of iterations.
  */
  void set_num_iterations(int num_iterations);

  /*!
    \brief Check if geo normalization is enabled.
    \return True if geo normalization is enabled, false otherwise.
  */
  bool get_enable_geo_norm_calculation() const { return do_geo; }

  /*!
    \brief Enable or disable geo normalization.
    \param do_geo True to enable geo normalization, false to disable.
  */
  void set_enable_geo_norm_calculation(bool do_geo);

  /*!
    \brief Check if block normalization is enabled.
    \return True if block normalization is enabled, false otherwise.
  */
  bool get_enable_block_norm_calculation() const { return do_block; }

  /*!
    \brief Enable or disable block normalization.
    \param do_block True to enable block normalization, false to disable.
  */
  void set_enable_block_norm_calculation(bool do_block);

  /*!
    \brief Check if symmetry per block is enabled.
    \return True if symmetry per block is enabled, false otherwise.
  */
  bool get_enable_symmetry_per_block() const { return do_symmetry_per_block; }

  /*!
    \brief Enable or disable symmetry per block.
    \param do_symmetry_per_block True to enable symmetry per block, false to disable.
  */
  void set_enable_symmetry_per_block(bool do_symmetry_per_block);

  /*!
    \brief Check if KL divergence calculation is enabled.
    This does not impact the calculation of the normalisation factors, only the display of the KL value.
    \return True if KL divergence calculation is enabled, false otherwise.
  */
  bool get_do_kl_calculation() const { return do_KL; }

  /*!
    \brief Enable or disable KL divergence calculation.
    This does not impact the calculation of the normalisation factors, it only prints the KL value to console.
    \param do_kl True to enable KL divergence calculation, false to disable.
  */
  void set_do_kl_calculation(bool do_kl);

  /*!
    \brief Check if display is enabled.
    \return True if display is enabled, false otherwise.
  */
  bool get_write_display_data() const { return do_display; }

  /*!
    \brief Enable or disable display.
    \param do_display True to enable display, false to disable.
  */
  void set_write_display_data(bool do_display);

  /*!
    \brief Check if saving to file is enabled.
    \return True if saving to file is enabled, false otherwise.
  */
  bool get_write_intermediates_to_file() const { return do_save_to_file; }

  /*!
    \brief Enable or disable saving to file.
    \param do_save_to_file True to enable saving to file, false to disable.
  */
  void set_write_intermediates_to_file(bool do_save_to_file);

private:
  /*!
   \brief Performs an efficiency iteration to update the efficiancies from the data_fan_sums and model.
   Additionally, handles the saving of the efficiencies iteration to file, KL calculation and display.
   \param[in] iter_num The iteration number
   \param[in] eff_iter_num The efficiency iteration number
   */
  void efficiency_iteration(int iter_num, int eff_iter_num);

  /*!
   \brief Performs a geo normalization iteration to update the geo data from the measured_geo_data and model_data.
   Additionally, handles the saving of the geo data iteration to file, KL calculation and display.
   \param[in] iter_num The iteration number
   */
  void geo_normalization_iteration(int iter_num);

  /*!
   \brief Performs a block normalization iteration to update the block data from the measured_block_data and model_data.
    Additionally, handles the saving of the block data iteration to file, KL calculation and display.
   * @param iter_num The iteration number
   */
  void block_normalization_iteration(int iter_num);

  /*!
    \brief Write the efficiencies to a file (regardless of the value of do_save_to_file)
    \param[in] iter_num The iteration number
    \param[in] eff_iter_num The efficiency iteration number
  */
  void write_efficiencies_to_file(int iter_num, int eff_iter_num) const;

  /*!
    \brief Write the efficiencies to a file (regardless of the value of do_save_to_file)
    \param[in] iter_num The iteration number
  */
  void write_geo_data_to_file(int iter_num) const;

  /*!
    \brief Write the efficiencies to a file (regardless of the value of do_save_to_file)
    \param[in] iter_num The iteration number
  */
  void write_block_data_to_file(int iter_num) const;

  /*!
   \brief Computes the threshold for the Kullback-Leibler divergence calculation. This is a purely heuristic value.
        \return The threshold value
   */
  float compute_threshold_for_KL();

  // Constructor parameters
  //! The prefix for the output files
  std::string out_filename_prefix;
  //! The number of (sub-)efficiency iterations to perform per iteration of the algorithm
  int num_eff_iterations;
  //! The number of algorithm iterations to perform
  int num_iterations;
  //! Whether to perform geo normalization calculations
  bool do_geo;
  //! Whether to perform block normalization calculations
  bool do_block;
  //! Whether to perform block normalization calculations with symmetry
  bool do_symmetry_per_block;
  //! Whether to perform Kullback-Leibler divergence calculations and display the KL value. This can be slow.
  bool do_KL;
  //! Whether to display the progress of the algorithm.
  bool do_display;
  //! Whether to save each iteration of the efficiencies, geo data and block data to file.
  //! This will only work if the out_filename_prefix is set.
  bool do_save_to_file;

  //! The projdata info of the measured data
  std::shared_ptr<const ProjDataInfo> projdata_info;

  // Calculated variables
  //! The efficiencies calculated by the algorithm
  DetectorEfficiencies norm_efficiencies;
  //! The geo data calculated by the algorithm, if do_geo is true
  BlockData3D norm_block_data;
  //! The block data calculated by the algorithm, if do_block is true
  GeoData3D norm_geo_data;

  //! The threshold for the Kullback-Leibler divergence calculation
  float threshold_for_KL;
  FanProjData model_fan_data;
  FanProjData fan_data;
  DetectorEfficiencies data_fan_sums;
  BlockData3D measured_block_data;
  GeoData3D measured_geo_data;

  //! Boolean to check if the data has been processed, see has_data_been_processed()
  bool data_processed = false;

  // do_KL specific varaibles
  FanProjData measured_fan_data;
  DetectorEfficiencies fan_sums;
  GeoData3D geo_data;
  BlockData3D block_data;
};

END_NAMESPACE_STIR
