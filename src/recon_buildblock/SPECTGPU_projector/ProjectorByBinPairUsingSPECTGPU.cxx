//
//
/*!
  \file
  \ingroup projection
  \ingroup SPECTGPU

  \brief non-inline implementations for stir::ProjectorByBinPairUsingSPECTGPU

  \author Daniel Deidda

*/
/*
    Copyright (C) 2026, National Physical Laboratory
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include "stir/recon_buildblock/SPECTGPU_projector/ProjectorByBinPairUsingSPECTGPU.h"
#include "stir/recon_buildblock/SPECTGPU_projector/ForwardProjectorByBinSPECTGPU.h"
#include "stir/recon_buildblock/SPECTGPU_projector/BackProjectorByBinSPECTGPU.h"

START_NAMESPACE_STIR

const char* const ProjectorByBinPairUsingSPECTGPU::registered_name = "SPECTGPU";

void
ProjectorByBinPairUsingSPECTGPU::initialise_keymap()
{
  base_type::initialise_keymap();
  parser.add_start_key("Projector Pair Using SPECTGPU Parameters");
  parser.add_stop_key("End Projector Pair Using SPECTGPU Parameters");
  parser.add_key("number of PSF sigmas", &_num_sigmas);
  parser.add_key("collimator slope", &_slope);
  parser.add_key("collimator sigma 0(cm)", &_sigma0);
  parser.add_key("attenuation image filename", &_att_filename);
  parser.add_key("verbosity", &_verbosity);
  parser.add_key("use_truncation", &_use_truncation);
}

void
ProjectorByBinPairUsingSPECTGPU::set_defaults()
{
  base_type::set_defaults();
  this->set_verbosity(true);
  this->set_use_truncation(false);
  this->_sigma0 = -1;
  this->_slope = -1;
  this->_num_sigmas = 2;
}

bool
ProjectorByBinPairUsingSPECTGPU::post_processing()
{
  this->set_verbosity(this->_verbosity);
  this->set_use_truncation(this->_use_truncation);

  set_sigma0(_sigma0);
  set_num_sigmas(_num_sigmas);
  set_slope(_slope);
  set_attenuation_filename(_att_filename);

  if (base_type::post_processing())
    return true;
  return false;
}

ProjectorByBinPairUsingSPECTGPU::ProjectorByBinPairUsingSPECTGPU()
{
  this->forward_projector_sptr.reset(new ForwardProjectorByBinSPECTGPU);
  this->back_projector_sptr.reset(new BackProjectorByBinSPECTGPU);
  set_defaults();
}

void
ProjectorByBinPairUsingSPECTGPU::set_sigma0(const float sigma0)
{
  _sigma0 = sigma0;

  auto fwd = dynamic_pointer_cast<ForwardProjectorByBinSPECTGPU>(this->forward_projector_sptr);

  auto bwd = dynamic_pointer_cast<BackProjectorByBinSPECTGPU>(this->back_projector_sptr);

  if (fwd)
    fwd->set_sigma0(_sigma0);

  if (bwd)
    bwd->set_sigma0(_sigma0);
}

void
ProjectorByBinPairUsingSPECTGPU::set_num_sigmas(const float num_sigmas)
{
  _num_sigmas = num_sigmas;

  auto fwd = dynamic_pointer_cast<ForwardProjectorByBinSPECTGPU>(this->forward_projector_sptr);

  auto bwd = dynamic_pointer_cast<BackProjectorByBinSPECTGPU>(this->back_projector_sptr);

  if (fwd)
    fwd->set_num_sigmas(_num_sigmas);

  if (bwd)
    bwd->set_num_sigmas(_num_sigmas);
}

void
ProjectorByBinPairUsingSPECTGPU::set_slope(const float slope)
{
  _slope = slope;

  auto fwd = dynamic_pointer_cast<ForwardProjectorByBinSPECTGPU>(this->forward_projector_sptr);

  auto bwd = dynamic_pointer_cast<BackProjectorByBinSPECTGPU>(this->back_projector_sptr);

  if (fwd)
    fwd->set_slope(_slope);

  if (bwd)
    bwd->set_slope(_slope);
}

void
ProjectorByBinPairUsingSPECTGPU::set_attenuation_filename(const std::string& filename)
{
  _att_filename = filename;

  auto fwd = dynamic_pointer_cast<ForwardProjectorByBinSPECTGPU>(this->forward_projector_sptr);

  auto bwd = dynamic_pointer_cast<BackProjectorByBinSPECTGPU>(this->back_projector_sptr);

  if (fwd)
    fwd->set_attenuation_filename(_att_filename);

  if (bwd)
    bwd->set_attenuation_filename(_att_filename);
}

void
ProjectorByBinPairUsingSPECTGPU::set_verbosity(const bool verbosity)
{
  _verbosity = verbosity;

  shared_ptr<ForwardProjectorByBinSPECTGPU> fwd_prj_downcast_sptr
      = dynamic_pointer_cast<ForwardProjectorByBinSPECTGPU>(this->forward_projector_sptr);
  if (fwd_prj_downcast_sptr)
    fwd_prj_downcast_sptr->set_verbosity(_verbosity);

  shared_ptr<BackProjectorByBinSPECTGPU> bck_prj_downcast_sptr
      = dynamic_pointer_cast<BackProjectorByBinSPECTGPU>(this->back_projector_sptr);
  if (bck_prj_downcast_sptr)
    bck_prj_downcast_sptr->set_verbosity(_verbosity);
}

void
ProjectorByBinPairUsingSPECTGPU::set_use_truncation(const bool use_truncation)
{
  _use_truncation = use_truncation;

  shared_ptr<ForwardProjectorByBinSPECTGPU> fwd_prj_downcast_sptr
      = dynamic_pointer_cast<ForwardProjectorByBinSPECTGPU>(this->forward_projector_sptr);
  if (fwd_prj_downcast_sptr)
    fwd_prj_downcast_sptr->set_use_truncation(_use_truncation);

  shared_ptr<BackProjectorByBinSPECTGPU> bck_prj_downcast_sptr
      = dynamic_pointer_cast<BackProjectorByBinSPECTGPU>(this->back_projector_sptr);
  if (bck_prj_downcast_sptr)
    bck_prj_downcast_sptr->set_use_truncation(_use_truncation);
}

END_NAMESPACE_STIR
