//
// $Id$
//
/*!

  \file

  \brief Instantiations of RegisteredObject

  Currently only necessary for VC

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#ifdef _MSC_VER
#pragma message("instantiating RegisteredObject<SinglesRates>")
#include "local/stir/SinglesRates.h"
// add here all roots of hierarchies based on RegisteredObject

#pragma message("instantiating RegisteredObject<RigidObject3DMotion>")
#include "local/stir/motion/RigidObject3DMotion.h"


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
