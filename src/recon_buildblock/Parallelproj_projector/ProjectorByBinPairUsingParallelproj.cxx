/*!
  \file
  \ingroup Parallelproj

  \brief non-inline implementations for stir::ProjectorByBinPairUsingParallelproj
  
  \author Richard Brown
  \author Kris Thielemans
    
*/
/*
    Copyright (C) 2019, 2021 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/


#include "stir/recon_buildblock/Parallelproj_projector/ProjectorByBinPairUsingParallelproj.h"
#include "stir/recon_buildblock/Parallelproj_projector/ForwardProjectorByBinParallelproj.h"
#include "stir/recon_buildblock/Parallelproj_projector/BackProjectorByBinParallelproj.h"
#include "stir/recon_buildblock/Parallelproj_projector/ParallelprojHelper.h"
#include "stir/Succeeded.h"

START_NAMESPACE_STIR


const char * const 
ProjectorByBinPairUsingParallelproj::registered_name =
  "Parallelproj";


void 
ProjectorByBinPairUsingParallelproj::initialise_keymap()
{
  base_type::initialise_keymap();
  parser.add_start_key("Projector Pair Using Parallelproj Parameters");
  parser.add_stop_key("End Projector Pair Using Parallelproj Parameters");
  parser.add_key("verbosity",&_verbosity);
  parser.add_key("restrict to cylindrical FOV", &_restrict_to_cylindrical_FOV);
}


void
ProjectorByBinPairUsingParallelproj::set_defaults()
{
  base_type::set_defaults();
  this->set_verbosity(true);
  this->set_restrict_to_cylindrical_FOV(true);
  this->_already_set_up = false;
}

bool
ProjectorByBinPairUsingParallelproj::post_processing()
{
    this->set_verbosity(this->_verbosity);

  if (base_type::post_processing())
    return true;
  return false;
}

ProjectorByBinPairUsingParallelproj::
ProjectorByBinPairUsingParallelproj()
{
  this->forward_projector_sptr.reset(new ForwardProjectorByBinParallelproj);
  this->back_projector_sptr.reset(new BackProjectorByBinParallelproj);
  set_defaults();
}

bool
ProjectorByBinPairUsingParallelproj::
get_restrict_to_cylindrical_FOV() const
{
  return this->_restrict_to_cylindrical_FOV;
}

void
ProjectorByBinPairUsingParallelproj::
set_restrict_to_cylindrical_FOV(bool val)
{
  this->_already_set_up = this->_already_set_up && (this->_restrict_to_cylindrical_FOV == val);
  this->_restrict_to_cylindrical_FOV = val;
}

Succeeded
ProjectorByBinPairUsingParallelproj::
set_up(const shared_ptr<const ProjDataInfo>& proj_data_info_sptr,
       const shared_ptr<const DiscretisedDensity<3,float> >& image_info_sptr)
{
  auto fwd_prj_downcast_sptr =
    dynamic_pointer_cast<ForwardProjectorByBinParallelproj>(this->forward_projector_sptr);
  if (!fwd_prj_downcast_sptr)
    error("internal error: forward projector should be ParallelProj");

  auto bck_prj_downcast_sptr =
    dynamic_pointer_cast<BackProjectorByBinParallelproj>(this->back_projector_sptr);
  if (!bck_prj_downcast_sptr)
    error("internal error: back projector should be ParallelProj");

  bck_prj_downcast_sptr->set_restrict_to_cylindrical_FOV(this->_restrict_to_cylindrical_FOV);
  fwd_prj_downcast_sptr->set_restrict_to_cylindrical_FOV(this->_restrict_to_cylindrical_FOV);
  this->_helper = std::make_shared<detail::ParallelprojHelper>(*proj_data_info_sptr, *image_info_sptr);
  fwd_prj_downcast_sptr->set_helper(this->_helper);
  bck_prj_downcast_sptr->set_helper(this->_helper);

  // the forward_projector->set_up etc will be called in the base class

  if (base_type::set_up(proj_data_info_sptr, image_info_sptr) != Succeeded::yes)
    return Succeeded::no;

  this->_already_set_up = true;
  return Succeeded::yes;
}

void ProjectorByBinPairUsingParallelproj::set_verbosity(const bool verbosity)
{
    _verbosity = verbosity;

    shared_ptr<ForwardProjectorByBinParallelproj> fwd_prj_downcast_sptr =
            dynamic_pointer_cast<ForwardProjectorByBinParallelproj>(this->forward_projector_sptr);
    if (fwd_prj_downcast_sptr)
        fwd_prj_downcast_sptr->set_verbosity(_verbosity);

    shared_ptr<BackProjectorByBinParallelproj> bck_prj_downcast_sptr =
            dynamic_pointer_cast<BackProjectorByBinParallelproj>(this->back_projector_sptr);
    if (bck_prj_downcast_sptr)
        bck_prj_downcast_sptr->set_verbosity(_verbosity);
}


END_NAMESPACE_STIR
