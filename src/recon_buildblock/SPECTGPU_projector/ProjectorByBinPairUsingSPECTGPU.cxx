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
  parser.add_key("verbosity", &_verbosity);
  parser.add_key("use_truncation", &_use_truncation);
}

void
ProjectorByBinPairUsingSPECTGPU::set_defaults()
{
  base_type::set_defaults();
  this->set_verbosity(true);
  this->set_use_truncation(false);
}

bool
ProjectorByBinPairUsingSPECTGPU::post_processing()
{
  this->set_verbosity(this->_verbosity);
  this->set_use_truncation(this->_use_truncation);

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
