//
//
/*!

  \file

  \brief Instantiations of RegisteredObject

  Currently only necessary for VC 6.0

  \author Kris Thielemans

*/
/*
    Copyright (C) 2000- 2004, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/
// note: include has to be before #ifdef as it's in this file that
// __STIR_REGISTRY_NOT_INLINE is defined
#include "stir/RegisteredObject.h"

#ifdef __STIR_REGISTRY_NOT_INLINE

#pragma message("instantiating RegisteredObject<SinglesRates>")
#include "stir_experimental/SinglesRates.h"
// add here all roots of hierarchies based on RegisteredObject

#pragma message("instantiating RegisteredObject<RigidObject3DMotion>")
#include "stir_experimental/motion/RigidObject3DMotion.h"


START_NAMESPACE_STIR

template <typename Root>
RegisteredObject<Root>::RegistryType& 
RegisteredObject<Root>::registry ()
{
  static RegistryType the_registry("None", 0);
  return the_registry;
}


//template RegisteredObject<SinglesRates>;

template RegisteredObject<RigidObject3DMotion>;
// add here all roots of hierarchies based on RegisteredObject


END_NAMESPACE_STIR

#endif
