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

START_NAMESPACE_STIR

class ProjData;

/*!
 \brief Find normalisation factors using a maximum likelihood approach

 Output is currently a set of text files in an (awkard) format, as used
 by \c apply_normfactors3D.

 \param[in] do_geo find geometric component
 \param[in] do_block estimate block-pair components (intended for timing alignment). Do NOT use.
 \param[in] do_symmetry_per_block estimate rotational symmetry per block if \c true, or per bucket (recommended) if \c false
 \param[in] use_lm_cache read fan-sums of the measured data directly (experimental)
 \param[in] use_model_fan_data read model fan data from binary file (experimental)
 \param[in] model_fan_data_filename filename to read

 \ingroup recon_buildblock
*/
void ML_estimate_component_based_normalisation(const std::string& out_filename_prefix,
                                               const ProjData& measured_data,
                                               const ProjData& model_data,
                                               int num_eff_iterations,
                                               int num_iterations,
                                               bool do_geo,
                                               bool do_block = false,
                                               bool do_symmetry_per_block = false,
                                               bool do_KL = false,
                                               bool do_display = false,
                                               bool use_lm_cache = false,
                                               bool use_model_fan_data = false,
                                               std::string model_fan_data_filename = "");

END_NAMESPACE_STIR
