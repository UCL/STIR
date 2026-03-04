//
//
/*!

  \file
  \ingroup projection
  \ingroup SPECTGPU

  \brief non-inline implementations for stir::ForwardProjectorByBinSPECTGPU

  \author Daniel Deidda


*/
/*
    Copyright (C) 2026, National Physical Laboratory
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include "stir/recon_buildblock/SPECTGPU_projector/ForwardProjectorByBinSPECTGPU.h"
#include "stir/recon_buildblock/SPECTGPU_projector/SPECTGPUHelper.h"
#include "stir/ProjDataInMemory.h"
#include "stir/RelatedViewgrams.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/recon_buildblock/TrivialDataSymmetriesForBins.h"
#include "stir/recon_array_functions.h"
#include "stir/error.h"

START_NAMESPACE_STIR

//////////////////////////////////////////////////////////
const char* const ForwardProjectorByBinSPECTGPU::registered_name = "SPECTGPU";

ForwardProjectorByBinSPECTGPU::ForwardProjectorByBinSPECTGPU()
    : _cuda_device(0),
      _cuda_verbosity(true),
      _use_truncation(false)
{
  this->_already_set_up = false;
}

ForwardProjectorByBinSPECTGPU::~ForwardProjectorByBinSPECTGPU()
{}

void
ForwardProjectorByBinSPECTGPU::initialise_keymap()
{
  parser.add_start_key("Forward Projector Using SPECTGPU Parameters");
  parser.add_stop_key("End Forward Projector Using SPECTGPU Parameters");
  parser.add_key("CUDA device", &_cuda_device);
  parser.add_key("verbosity", &_cuda_verbosity);
}

void
ForwardProjectorByBinSPECTGPU::set_up(const shared_ptr<const ProjDataInfo>& proj_data_info_sptr,
                                      const shared_ptr<const DiscretisedDensity<3, float>>& density_info_sptr)
{
  ForwardProjectorByBin::set_up(proj_data_info_sptr, density_info_sptr);
  check(*proj_data_info_sptr, *_density_sptr);
  _symmetries_sptr.reset(new TrivialDataSymmetriesForBins(proj_data_info_sptr));

  
  // Initialise projected_data_sptr from this->_proj_data_info_sptr
  _projected_data_sptr.reset(new ProjDataInMemory(this->_density_sptr->get_exam_info_sptr(), proj_data_info_sptr));

  // Set up the SPECTGPU binary helper
  _helper.set_scanner_type(proj_data_info_sptr->get_scanner_ptr()->get_type());
  _helper.set_cuda_device_id(_cuda_device);
  _helper.set_att(0);
  _helper.set_verbose(_cuda_verbosity);
  _helper.set_up();
}


void
ForwardProjectorByBinSPECTGPU::actual_forward_project(
    RelatedViewgrams<float>&, const DiscretisedDensity<3, float>&, const int, const int, const int, const int)
{
  throw std::runtime_error("Need to use set_input() if wanting to use ForwardProjectorByBinSPECTGPU.");
}

void
ForwardProjectorByBinSPECTGPU::actual_forward_project(
    RelatedViewgrams<float>& viewgrams, const int, const int, const int, const int)
{
  //    if (min_axial_pos_num != _proj_data_info_sptr->get_min_axial_pos_num() ||
  //         ... )
  //       error();

  viewgrams = _projected_data_sptr->get_related_viewgrams(viewgrams.get_basic_view_segment_num(), _symmetries_sptr);
}

void
ForwardProjectorByBinSPECTGPU::set_input(const DiscretisedDensity<3, float>& density)
{
  ForwardProjectorByBin::set_input(density);

  // Before forward projection, we enforce a truncation outside of the FOV.
  // This is because the SPECTGPU FOV is smaller than the STIR FOV and this
  // could cause some voxel values to spiral out of control.
  if (_use_truncation)
    truncate_rim(*_density_sptr, 17);

  // --------------------------------------------------------------- //
  //   STIR -> SPECTGPU image data conversion
  // --------------------------------------------------------------- //

  std::vector<float> np_vec = _helper.create_SPECTGPU_image();
  _helper.convert_image_stir_to_SPECTGPU(np_vec, *_density_sptr);

  // --------------------------------------------------------------- //
  //   Forward projection
  // --------------------------------------------------------------- //

  std::vector<float> sino = _helper.create_SPECTGPU_sinogram();
  _helper.forward_project(sino, np_vec);

  std::vector<float> sino = _helper.create_SPECTGPU_sinogram();
  

  // --------------------------------------------------------------- //
  //   SPECTGPU -> STIR projection data conversion
  // --------------------------------------------------------------- //

  _helper.convert_proj_data_SPECTGPU_to_stir(*_projected_data_sptr, sino);
}

END_NAMESPACE_STIR
