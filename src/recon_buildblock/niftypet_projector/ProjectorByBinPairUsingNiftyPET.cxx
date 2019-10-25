//
//
/*!
  \file
  \ingroup projection

  \brief non-inline implementations for stir::ProjectorByBinPairUsingNiftyPET
  
  \author Richard Brown
    
*/
/*
    Copyright (C) 2000- 2011, Hammersmith Imanet Ltd
    Copyright (C) 2018, University College London
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for details
*/


#include "stir/recon_buildblock/niftypet_projector/ProjectorByBinPairUsingNiftyPET.h"
#include "stir/recon_buildblock/niftypet_projector/ForwardProjectorByBinNiftyPET.h"
#include "stir/recon_buildblock/niftypet_projector/BackProjectorByBinNiftyPET.h"

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
}


void
ProjectorByBinPairUsingNiftyPET::set_defaults()
{
  base_type::set_defaults();
  this->_verbosity = true;
}

bool
ProjectorByBinPairUsingNiftyPET::post_processing()
{
  if (base_type::post_processing())
    return true;

  this->forward_projector_sptr.reset(new ForwardProjectorByBinNiftyPET);
  this->back_projector_sptr.reset(new BackProjectorByBinNiftyPET);
  return false;
}

ProjectorByBinPairUsingNiftyPET::
ProjectorByBinPairUsingNiftyPET()
{
  set_defaults();
}

Succeeded
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
}

END_NAMESPACE_STIR
