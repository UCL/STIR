//
//
/*!
  \file
  \ingroup projection
  \ingroup NiftyPET

  \brief non-inline implementations for stir::ProjectorByBinPairUsingNiftyPET
  
  \author Richard Brown
    
*/
/*
    Copyright (C) 2019, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/


#include "stir/recon_buildblock/NiftyPET_projector/ProjectorByBinPairUsingNiftyPET.h"
#include "stir/recon_buildblock/NiftyPET_projector/ForwardProjectorByBinNiftyPET.h"
#include "stir/recon_buildblock/NiftyPET_projector/BackProjectorByBinNiftyPET.h"

START_NAMESPACE_STIR


const char * const 
ProjectorByBinPairUsingNiftyPET::registered_name =
  "NiftyPET";


void 
ProjectorByBinPairUsingNiftyPET::initialise_keymap()
{
  base_type::initialise_keymap();
  parser.add_start_key("Projector Pair Using NiftyPET Parameters");
  parser.add_stop_key("End Projector Pair Using NiftyPET Parameters");
  parser.add_key("verbosity",&_verbosity);
  parser.add_key("use_truncation",&_use_truncation);
}


void
ProjectorByBinPairUsingNiftyPET::set_defaults()
{
  base_type::set_defaults();
  this->set_verbosity(true);
  this->set_use_truncation(false);
}

bool
ProjectorByBinPairUsingNiftyPET::post_processing()
{
    this->set_verbosity(this->_verbosity);
    this->set_use_truncation(this->_use_truncation);

  if (base_type::post_processing())
    return true;
  return false;
}

ProjectorByBinPairUsingNiftyPET::
ProjectorByBinPairUsingNiftyPET()
{
  this->forward_projector_sptr.reset(new ForwardProjectorByBinNiftyPET);
  this->back_projector_sptr.reset(new BackProjectorByBinNiftyPET);
  set_defaults();
}

/*Succeeded
ProjectorByBinPairUsingNiftyPET::
set_up(const shared_ptr<ProjDataInfo>& proj_data_info_sptr,
       const shared_ptr<DiscretisedDensity<3,float> >& image_info_sptr)
{    	 
  // proj_matrix_sptr->set_up()  not needed as the projection matrix will be set_up indirectly by
  // the forward_projector->set_up (which is called in the base class)
  // proj_matrix_sptr->set_up(proj_data_info_sptr, image_info_sptr);

  if (base_type::set_up(proj_data_info_sptr, image_info_sptr) != Succeeded::yes)
    return Succeeded::no;

  return Succeeded::yes;
}*/

void ProjectorByBinPairUsingNiftyPET::set_verbosity(const bool verbosity)
{
    _verbosity = verbosity;

    shared_ptr<ForwardProjectorByBinNiftyPET> fwd_prj_downcast_sptr =
            dynamic_pointer_cast<ForwardProjectorByBinNiftyPET>(this->forward_projector_sptr);
    if (fwd_prj_downcast_sptr)
        fwd_prj_downcast_sptr->set_verbosity(_verbosity);

    shared_ptr<BackProjectorByBinNiftyPET> bck_prj_downcast_sptr =
            dynamic_pointer_cast<BackProjectorByBinNiftyPET>(this->back_projector_sptr);
    if (bck_prj_downcast_sptr)
        bck_prj_downcast_sptr->set_verbosity(_verbosity);
}

void ProjectorByBinPairUsingNiftyPET::set_use_truncation(const bool use_truncation)
{
    _use_truncation = use_truncation;

    shared_ptr<ForwardProjectorByBinNiftyPET> fwd_prj_downcast_sptr =
            dynamic_pointer_cast<ForwardProjectorByBinNiftyPET>(this->forward_projector_sptr);
    if (fwd_prj_downcast_sptr)
        fwd_prj_downcast_sptr->set_use_truncation(_use_truncation);

    shared_ptr<BackProjectorByBinNiftyPET> bck_prj_downcast_sptr =
            dynamic_pointer_cast<BackProjectorByBinNiftyPET>(this->back_projector_sptr);
    if (bck_prj_downcast_sptr)
        bck_prj_downcast_sptr->set_use_truncation(_use_truncation);
}

END_NAMESPACE_STIR
