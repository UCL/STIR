//
//
/*!
  \file
  \ingroup projection
  \ingroup SPECTGPU

  \brief wrapper functions for stir::ForwardProjectorByBinSPECTGPU

  \author Daniel Deidda

*/
/*
    Copyright (C) 2026, National Physical Laboratory
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include "stir/cuda_utilities.h"
#include "stir/RelatedViewgrams.h"
#include "stir/DiscretisedDensity.h"
#include "stir/recon_buildblock/SPECTGPU_projector/ForwardProjectorByBinSPECTGPU.h"

START_NAMESPACE_STIR

void run_forward_projection_cuda(RelatedViewgrams<float>& stir_sino,
                                 AllocatedStack& stack,
                                 bool do_atten,
                                 float coll_sigma0_cm,
                                 float num_sigmas,
                                 float coll_slope,
                                 int num_views,
                                 int min_ax,
                                 int max_ax,
                                 int min_tg,
                                 int max_tg,
                                 unsigned int block_x,
                                 unsigned int block_y,
                                 unsigned int block_z,
                                 unsigned int grid_x,
                                 unsigned int grid_y,
                                 unsigned int grid_z,
                                 float spacing_x,
                                 float spacing_y,
                                 float spacing_z,
                                 float origin_x,
                                 float origin_y,
                                 float origin_z,
                                 int dim_x,
                                 int dim_y,
                                 int dim_z,
                                 int min_z,
                                 int min_y,
                                 int min_x);
END_NAMESPACE_STIR
