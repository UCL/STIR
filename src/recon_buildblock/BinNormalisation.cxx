//
// $Id$
//
/*!
  \file
  \ingroup normalisation

  \brief Implementation for class BinNormalisation

  \author Kris Thielemans
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/


#include "stir/recon_buildblock/BinNormalisation.h"
#include "stir/Succeeded.h"

START_NAMESPACE_STIR

BinNormalisation::
~BinNormalisation()
{}

Succeeded
BinNormalisation::
set_up(const shared_ptr<ProjDataInfo>& )
{
  return Succeeded::yes;  
}

 
END_NAMESPACE_STIR

