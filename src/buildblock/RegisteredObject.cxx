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
