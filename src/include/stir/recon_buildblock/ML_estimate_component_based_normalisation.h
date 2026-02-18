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

  \ingroup recon_buildblock
*/
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

END_NAMESPACE_STIR
