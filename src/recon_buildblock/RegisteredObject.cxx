//
// $Id$
//
/*!

  \file
  \ingroup recon_buildblock
  \brief instantiations of RegisteredObject for classes in recon_buildblock
  (only useful for Microsoft Visual Studio)

  \author Kris Thielemans

  \date $Date$
  \version $Revision$
*/

#ifdef _MSC_VER
#pragma message("instantiating RegisteredObject<GeneralisedPrior<float> >")
#include "tomo/recon_buildblock/GeneralisedPrior.h"
// and others
START_NAMESPACE_TOMO

template <typename Root>
RegisteredObject<Root>::RegistryType& 
RegisteredObject<Root>::registry ()
{
  static RegistryType the_registry("None", 0);
  return the_registry;
}

template RegisteredObject<GeneralisedPrior<float> >;


END_NAMESPACE_TOMO

#endif
