//
// $Id$
//
/*!

  \file

  \brief Instantiations of RegisteredObject

  Currently only necessary for VC

  \author Kris Thielemans

  \date $Date$
  \version $Revision$
*/

#ifdef _MSC_VER
#pragma message("instantiating RegisteredObject<ImageProcessor<3,float> >")
#include "tomo/ImageProcessor.h"
// add here all roots of hierarchies based on RegisteredObject

START_NAMESPACE_TOMO

template <typename Root>
RegisteredObject<Root>::RegistryType& 
RegisteredObject<Root>::registry ()
{
  static RegistryType the_registry("None", 0);
  return the_registry;
}

template RegisteredObject<ImageProcessor<3,float> >;
// add here all roots of hierarchies based on RegisteredObject

END_NAMESPACE_TOMO

#endif
