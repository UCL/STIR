//
// $Id$
//
/*!
  \file
  \ingroup projection

  \brief non-inline implementations for stir::ProjectorByBinPairUsingSeparateProjectors
  
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


#include "stir/recon_buildblock/ProjectorByBinPairUsingSeparateProjectors.h"
#include "stir/is_null_ptr.h"
#include "stir/Succeeded.h"

START_NAMESPACE_STIR


const char * const 
ProjectorByBinPairUsingSeparateProjectors::registered_name =
  "Separate Projectors";


void 
ProjectorByBinPairUsingSeparateProjectors::initialise_keymap()
{
  parser.add_start_key("Projector Pair Using Separate Projectors Parameters");
  parser.add_stop_key("End Projector Pair Using Separate Projectors Parameters");
  parser.add_parsing_key("Forward projector type",&forward_projector_sptr);
  parser.add_parsing_key("Back projector type",&back_projector_sptr);
}


void
ProjectorByBinPairUsingSeparateProjectors::
set_defaults()
{
  base_type::set_defaults();
  forward_projector_sptr.reset();
  back_projector_sptr.reset();
}

bool
ProjectorByBinPairUsingSeparateProjectors::
post_processing()
{
  if (base_type::post_processing())
    return true;
  if (is_null_ptr(forward_projector_sptr))
  { warning("No valid forward projector is defined\n"); return true; }

  if (is_null_ptr(back_projector_sptr))
  { warning("No valid back projector is defined\n"); return true; }

  return false;
}

ProjectorByBinPairUsingSeparateProjectors::
ProjectorByBinPairUsingSeparateProjectors()
{
  set_defaults();
}

ProjectorByBinPairUsingSeparateProjectors::
ProjectorByBinPairUsingSeparateProjectors(const shared_ptr<ForwardProjectorByBin>& forward_projector_sptr_v,
                                          const shared_ptr<BackProjectorByBin>& back_projector_sptr_v)
{
  forward_projector_sptr = forward_projector_sptr_v;
  back_projector_sptr = back_projector_sptr_v;
}


END_NAMESPACE_STIR
