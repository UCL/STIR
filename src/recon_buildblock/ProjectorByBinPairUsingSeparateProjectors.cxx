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

START_NAMESPACE_STIR


const char * const 
ProjectorByBinPairUsingSeparateProjectors::registered_name =
  "Separate Projectors";


void 
ProjectorByBinPairUsingSeparateProjectors::initialise_keymap()
{
  parser.add_start_key("Projector Pair Using Separate Projectors Parameters");
  parser.add_stop_key("End Projector Pair Using Separate Projectors Parameters");
  parser.add_parsing_key("Forward projector type",&forward_projector_ptr);
  parser.add_parsing_key("Back projector type",&back_projector_ptr);
}


void
ProjectorByBinPairUsingSeparateProjectors::set_defaults()
{}

ProjectorByBinPairUsingSeparateProjectors::
ProjectorByBinPairUsingSeparateProjectors()
{
  set_defaults();
}

ProjectorByBinPairUsingSeparateProjectors::
ProjectorByBinPairUsingSeparateProjectors(const shared_ptr<ForwardProjectorByBin>& forward_projector_ptr_v,
                                          const shared_ptr<BackProjectorByBin>& back_projector_ptr_v)
{
  forward_projector_ptr = forward_projector_ptr_v;
  back_projector_ptr = back_projector_ptr_v;
}

END_NAMESPACE_STIR
