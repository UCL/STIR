//
//
/*!
  \file
  \ingroup projection
  \ingroup Parallelproj

  \brief non-inline implementations for stir::ProjectorByBinPairUsingParallelproj
  
  \author Richard Brown
  \author Kris Thielemans
    
*/
/*
    Copyright (C) 2019, 2021 University College London
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


#include "stir/recon_buildblock/Parallelproj_projector/ProjectorByBinPairUsingParallelproj.h"
#include "stir/recon_buildblock/Parallelproj_projector/ForwardProjectorByBinParallelproj.h"
#include "stir/recon_buildblock/Parallelproj_projector/BackProjectorByBinParallelproj.h"

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

/*Succeeded
ProjectorByBinPairUsingParallelproj::
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
