//
// $Id$
//
/*!
  \file
  \ingroup projection

  \brief non-inline implementations for ProjectorByBinPairUsingSeparateProjectors
  
  \author Kris Thielemans
    
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
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
  forward_projector_sptr = 0;
  back_projector_sptr = 0;
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
