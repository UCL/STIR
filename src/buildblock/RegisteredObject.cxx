//
//
/*!

  \file
  \ingroup buildblock
  \brief Instantiations of stir::RegisteredObject

  Currently only necessary for VC 6.0

  \author Kris Thielemans

*/
/*
    Copyright (C) 2000- 2009, Hammersmith  Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

// note: include has to be before #ifdef as it's in this file that
// __STIR_REGISTRY_NOT_INLINE is defined
#include "stir/RegisteredObject.h"

#ifdef __STIR_REGISTRY_NOT_INLINE
#pragma message("instantiating RegisteredObject<DataProcessor<DiscretisedDensity<3,float> > >")
#include "stir/DataProcessor.h"
#include "stir/DiscretisedDensity.h"

// add here all roots of hierarchies based on RegisteredObject

START_NAMESPACE_STIR

template <typename Root>
RegisteredObject<Root>::RegistryType& 
RegisteredObject<Root>::registry ()
{
  static RegistryType the_registry("None", 0);
  return the_registry;
}

template RegisteredObject<DataProcessor<DiscretisedDensity<3,float> > >;
// add here all roots of hierarchies based on RegisteredObject

END_NAMESPACE_STIR

#endif
