//
//
/*
    Copyright (C) 2000- 2007, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file

  \brief Instantiations of stir::RegisteredObject

  Currently only necessary for VC 6.0

  \author Kris Thielemans

*/

// note: include has to be before #ifdef as it's in this file that
// __STIR_REGISTRY_NOT_INLINE is defined
#include "stir/RegisteredObject.h"

#ifdef __STIR_REGISTRY_NOT_INLINE

#pragma message("instantiating RegisteredObject<OutputFileFormat >")
#include "stir/IO/OutputFileFormat.h"
// add here all roots of hierarchies based on RegisteredObject

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

template RegisteredObject<OutputFileFormat >;
// add here all roots of hierarchies based on RegisteredObject

END_NAMESPACE_STIR

#endif
