//
// $Id$
//
/*!
  \file
  \ingroup priors
  \brief  implementation of the GeneralisedPrior
    
  \author Kris Thielemans
  \author Sanida Mustafovic      
  $Date$  
  $Revision$
*/
/*
    Copyright (C) 2002- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/

#include "stir/recon_buildblock/GeneralisedPrior.h"

START_NAMESPACE_STIR


template <typename elemT>
void 
GeneralisedPrior<elemT>::initialise_keymap()
{
  parser.add_key("penalisation factor", &penalisation_factor); 
}


template <typename elemT>
void
GeneralisedPrior<elemT>::set_defaults()
{
  penalisation_factor = 0;  
}

#  ifdef _MSC_VER
// prevent warning message on instantiation of abstract class 
#  pragma warning(disable:4661)
#  endif

template class GeneralisedPrior<float>;

END_NAMESPACE_STIR

