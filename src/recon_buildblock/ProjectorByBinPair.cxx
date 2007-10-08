//
// $Id$
//
/*!
  \file
  \ingroup projection

  \brief non-inline implementations for ProjectorByBinPair
  
  \author Kris Thielemans
    
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/


#include "stir/recon_buildblock/ProjectorByBinPair.h"
#include "stir/Succeeded.h"

START_NAMESPACE_STIR


ProjectorByBinPair::
ProjectorByBinPair()
{
}

Succeeded
ProjectorByBinPair::
set_up(const shared_ptr<ProjDataInfo>& proj_data_info_ptr,
       const shared_ptr<DiscretisedDensity<3,float> >& image_info_ptr)

{    	 
  // TODO use return values
  forward_projector_sptr->set_up(proj_data_info_ptr, image_info_ptr);
  back_projector_sptr->set_up(proj_data_info_ptr, image_info_ptr);
  return Succeeded::yes;
}

//ForwardProjectorByBin const * 
const shared_ptr<ForwardProjectorByBin>
ProjectorByBinPair::
get_forward_projector_sptr() const
{
  return forward_projector_sptr;
}

//BackProjectorByBin const * 
const shared_ptr<BackProjectorByBin>
ProjectorByBinPair::
get_back_projector_sptr() const
{
  return back_projector_sptr;
}

END_NAMESPACE_STIR
