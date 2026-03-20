//
//
/*!
  \file
  \ingroup projection
  \ingroup SPECTGPU

  \brief non-inline implementations for stir::BackProjectorByBinSPECTGPU

  \author Daniel Deidda

*/
/*
    Copyright (C) 2026, National Physical Laboratory
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include "stir/recon_buildblock/SPECTGPU_projector/BackProjectorByBinSPECTGPU.h"
#include "stir/recon_buildblock/SPECTGPU_projector/SPECTGPUHelper.h"
#include "stir/DiscretisedDensity.h"
#include "stir/RelatedViewgrams.h"
#include "stir/recon_buildblock/TrivialDataSymmetriesForBins.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/recon_array_functions.h"
#include "stir/error.h"

START_NAMESPACE_STIR

//////////////////////////////////////////////////////////
const char* const BackProjectorByBinSPECTGPU::registered_name = "SPECTGPU";

BackProjectorByBinSPECTGPU::BackProjectorByBinSPECTGPU()
    : _cuda_device(0),
      _cuda_verbosity(true),
      _use_truncation(false)
{
  this->_already_set_up = false;
}

BackProjectorByBinSPECTGPU::~BackProjectorByBinSPECTGPU()
{}

void
BackProjectorByBinSPECTGPU::initialise_keymap()
{
  parser.add_start_key("Back Projector Using SPECTGPU Parameters");
  parser.add_stop_key("End Back Projector Using SPECTGPU Parameters");
  parser.add_key("CUDA device", &_cuda_device);
  parser.add_key("verbosity", &_cuda_verbosity);
}

void
BackProjectorByBinSPECTGPU::set_up(const shared_ptr<const ProjDataInfo>& proj_data_info_sptr,
                                   const shared_ptr<const DiscretisedDensity<3, float>>& density_info_sptr)
{
  BackProjectorByBin::set_up(proj_data_info_sptr, density_info_sptr);
  check(*proj_data_info_sptr, *_density_sptr);
  _symmetries_sptr.reset(new TrivialDataSymmetriesForBins(proj_data_info_sptr));


  // Set up the SPECTGPU binary helper
  _helper.set_cuda_device_id(_cuda_device);
  _helper.set_scanner_type(proj_data_info_sptr->get_scanner_ptr()->get_type());
  _helper.set_att(0);
  _helper.set_verbose(_cuda_verbosity);
  _helper.set_up();

  // Create sinogram
  _np_sino = _helper.create_SPECTGPU_sinogram();
}

void
BackProjectorByBinSPECTGPU::back_project(const ProjData& proj_data, int subset_num, int num_subsets)
{
  // Check the user has tried to project all data
  if (subset_num != 0 || num_subsets != 1)
    error("BackProjectorByBinSPECTGPU::back_project "
          "only works with all data (no subsets).");

  _helper.convert_proj_data_stir_to_SPECTGPU(_np_sino, proj_data);
}

void
BackProjectorByBinSPECTGPU::get_output(DiscretisedDensity<3, float>& density) const
{
  
  std::vector<float> sino = _helper.create_SPECTGPU_sinogram();

  // --------------------------------------------------------------- //
  //   Back project
  // --------------------------------------------------------------- //

  std::vector<float> np_im = _helper.create_SPECTGPU_image();
  _helper.back_project(np_im, sino_);

  // --------------------------------------------------------------- //
  //   SPECTGPU -> STIR image conversion
  // --------------------------------------------------------------- //

  _helper.convert_image_SPECTGPU_to_stir(density, np_im);

  // After the back projection, we enforce a truncation outside of the FOV.
  // This is because the SPECTGPU FOV is smaller than the STIR FOV and this
  // could cause some voxel values to spiral out of control.
  if (_use_truncation)
    truncate_rim(density, 17);
}

void
BackProjectorByBinSPECTGPU::start_accumulating_in_new_target()
{
  // Call base level
  BackProjectorByBin::start_accumulating_in_new_target();
  // Also reset the SPECTGPU sinogram
  _np_sino = _helper.create_SPECTGPU_sinogram();
}

void
BackProjectorByBinSPECTGPU::actual_back_project(DiscretisedDensity<3, float> &stir_image,
    const RelatedViewgrams<float>& related_viewgrams, const int, const int, const int, const int)
{

//    call the kernels for backward
}

END_NAMESPACE_STIR
