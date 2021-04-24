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
}


void
ProjectorByBinPairUsingParallelproj::set_defaults()
{
  base_type::set_defaults();
  this->set_verbosity(true);
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

Succeeded
ProjectorByBinPairUsingParallelproj::
set_up(const shared_ptr<ProjDataInfo>& proj_data_info_sptr,
       const shared_ptr<DiscretisedDensity<3,float> >& image_info_sptr)
{
  _helper = std::make_shared<detail::ParallelprojHelper>(*proj_data_info_sptr, *image_info_sptr);
  dynamic_pointer_cast<ForwardProjectorByBinParallelproj>(this->forward_projector_sptr)
    ->set_helper(_helper);
  dynamic_pointer_cast<BackProjectorByBinParallelproj>(this->back_projector_sptr)
    ->set_helper(_helper);

  // the forward_projector->set_up etc will be called in the base class

  if (base_type::set_up(proj_data_info_sptr, image_info_sptr) != Succeeded::yes)
    return Succeeded::no;

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
