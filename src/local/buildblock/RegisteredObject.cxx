//
// $Id$
//
/*!

  \file
  \ingroup buildblock
  \brief instantiations of RegisteredObject for classes in recon_buildblock
  (only useful for Microsoft Visual Studio)

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#ifdef __STIR_REGISTRY_NOT_INLINE

#pragma message("instantiating RegisteredObject<SinglesRates >")
#include "local/stir/SinglesRates.h"


// and others
START_NAMESPACE_STIR

template <typename Root>
RegisteredObject<Root>::RegistryType& 
RegisteredObject<Root>::registry ()
{
  static RegistryType the_registry("None", 0);
  return the_registry;
}

#  ifdef _MSC_VER
// prevent warning message on reinstantiation, 
// note that we get a linking error if we don't have the explicit instantiation below
#  pragma warning(disable:4660)
#  endif


template RegisteredObject<SinglesRates >;

END_NAMESPACE_STIR

#endif
