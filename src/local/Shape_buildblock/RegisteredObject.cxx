//
// $Id$
//
/*!

  \file
  \ingroup Shape3D
  \brief instantiations of RegisteredObject for classes in Shape_buildblock
  (only useful for Microsoft Visual Studio)

  \author Kris Thielemans

  \date $Date$
  \version $Revision$
*/

#ifdef _MSC_VER
#pragma message("instantiating RegisteredObject<Shape3D>")
#include "local/tomo/Shape/Shape3D.h"

// and others
START_NAMESPACE_TOMO

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

template RegisteredObject<Shape3D>; 

END_NAMESPACE_TOMO

#endif
