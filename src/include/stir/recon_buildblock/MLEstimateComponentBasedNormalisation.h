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
    \param out_filename_prefix The prefix for the output files
    \param measured_data The measured projection data
    \param model_data The model projection data
    \param num_eff_iterations_v The number of (sub-)efficiency iterations to perform per iteration of the algorithm
    \param num_iterations_v The number of algorithm iterations to perform
    \param do_geo_v Whether to perform geo normalization calculations
    \param do_block_v Whether to perform block normalization calculations
    \param do_symmetry_per_block_v Whether to perform block normalization calculations with symmetry
    \param do_KL_v Whether to perform Kullback-Leibler divergence calculations and display the KL value. This can be slow.
    \param do_display_v Whether to display the progress of the algorithm.
    \param do_save_to_file_v Whether to save the each iteration of the efficiencies, geo data and block data to file.
  */
  MLEstimateComponentBasedNormalisation(std::string out_filename_prefix,
                                        const ProjData& measured_data_v,
                                        const ProjData& model_data_v,
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
  bool has_processed_data() const;

  //! Get the efficiencies, nullptr if not calculated
  std::shared_ptr<DetectorEfficiencies> get_efficiencies() const;
  //! Get the geo data, nullptr if not calculated
  std::shared_ptr<GeoData3D> get_geo_data() const;
  //! Get the block data, nullptr if not calculated
  std::shared_ptr<BlockData3D> get_block_data() const;

  BinNormalisationPETFromComponents construct_bin_normfactors_from_components() const;

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
  //! Whether to save the each iteration of the efficiencies, geo data and block data to file.
  bool do_save_to_file;

  // The projdata info of the measured data
  std::shared_ptr<const ProjDataInfo> projdata_info;

  // Calculated variables
  std::shared_ptr<DetectorEfficiencies> efficiencies_ptr;
  std::shared_ptr<BlockData3D> norm_block_data_ptr;
  std::shared_ptr<GeoData3D> norm_geo_data_ptr;
  //! The threshold for the Kullback-Leibler divergence calculation
  float threshold_for_KL;
  FanProjData model_fan_data;
  FanProjData fan_data;
  DetectorEfficiencies data_fan_sums;
  BlockData3D measured_block_data;
  GeoData3D measured_geo_data;

  bool data_processed = false;

  // do_KL specific varaibles
  FanProjData measured_fan_data;
  DetectorEfficiencies fan_sums;
  GeoData3D geo_data;
  BlockData3D block_data;
};

END_NAMESPACE_STIR
