//
// $Id$
//
/*!
  \file
  \ingroup recon_buildblock

  \brief non-inline implementations for ProjectorByBinPairUsingSeparateProjectors
  
  \author Kris Thielemans
    
  $Date$
  $Revision$
*/


#include "recon_buildblock/ProjectorByBinPairUsingSeparateProjectors.h"

START_NAMESPACE_TOMO


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

END_NAMESPACE_TOMO
