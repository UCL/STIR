//
// $Id$
//
/*!
  \file
  \ingroup projection

  \brief non-inline implementations for stir::ProjectorByBinPairUsingProjMatrixByBin
  
  \author Kris Thielemans
    
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
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


#include "stir/recon_buildblock/ProjectorByBinPairUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/ForwardProjectorByBinUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/BackProjectorByBinUsingProjMatrixByBin.h"
#include "stir/is_null_ptr.h"
#include "stir/Succeeded.h"

START_NAMESPACE_STIR


const char * const 
ProjectorByBinPairUsingProjMatrixByBin::registered_name =
  "Matrix";


void 
ProjectorByBinPairUsingProjMatrixByBin::initialise_keymap()
{
  base_type::initialise_keymap();
  parser.add_start_key("Projector Pair Using Matrix Parameters");
  parser.add_stop_key("End Projector Pair Using Matrix Parameters");
  parser.add_parsing_key("Matrix type",&proj_matrix_sptr);
}


void
ProjectorByBinPairUsingProjMatrixByBin::set_defaults()
{
  base_type::set_defaults();
  this->proj_matrix_sptr.reset();
}

bool
ProjectorByBinPairUsingProjMatrixByBin::post_processing()
{
  if (base_type::post_processing())
    return true;
  if (is_null_ptr(proj_matrix_sptr))
    { warning("No valid projection matrix is defined\n"); return true; }
  return false;
}

ProjectorByBinPairUsingProjMatrixByBin::
ProjectorByBinPairUsingProjMatrixByBin()
{
  set_defaults();
}

ProjectorByBinPairUsingProjMatrixByBin::
ProjectorByBinPairUsingProjMatrixByBin(  
    const shared_ptr<ProjMatrixByBin>& proj_matrix_sptr)	   
    : proj_matrix_sptr(proj_matrix_sptr)
{}

Succeeded
ProjectorByBinPairUsingProjMatrixByBin::
set_up(const shared_ptr<ProjDataInfo>& proj_data_info_sptr,
       const shared_ptr<DiscretisedDensity<3,float> >& image_info_sptr)
{    	 

  this->forward_projector_sptr.reset(new ForwardProjectorByBinUsingProjMatrixByBin(proj_matrix_sptr));
  this->back_projector_sptr.reset(new BackProjectorByBinUsingProjMatrixByBin(proj_matrix_sptr));

  // proj_matrix_sptr->set_up()  not needed as the projection matrix will be set_up indirectly by
  // the forward_projector->set_up (which is called in the base class)
  // proj_matrix_sptr->set_up(proj_data_info_sptr, image_info_sptr);

  if (base_type::set_up(proj_data_info_sptr, image_info_sptr) != Succeeded::yes)
    return Succeeded::no;

  return Succeeded::yes;
}

ProjMatrixByBin const * 
ProjectorByBinPairUsingProjMatrixByBin::
get_proj_matrix_ptr() const
{
  return proj_matrix_sptr.get();
}

END_NAMESPACE_STIR
