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

#pragma message("instantiating RegisteredObject<ProjMatrixByBin>")
#include "recon_buildblock/ProjMatrixByBin.h"

#pragma message("instantiating RegisteredObject<ProjectorByBinPair>")
#include "recon_buildblock/ProjectorByBinPair.h"

#pragma message("instantiating RegisteredObject<ForwardProjectorByBin>")
#include "recon_buildblock/ForwardProjectorByBin.h"

#pragma message("instantiating RegisteredObject<BackProjectorByBin>")
#include "recon_buildblock/BackProjectorByBin.h"

#pragma message("instantiating RegisteredObject<BinNormalisation>")
#include "tomo/recon_buildblock/BinNormalisation.h"

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


template RegisteredObject<GeneralisedPrior<float> >;

template RegisteredObject<ProjMatrixByBin>; 

template RegisteredObject<ProjectorByBinPair>;

template RegisteredObject<ForwardProjectorByBin>;

template RegisteredObject<BackProjectorByBin>;

template RegisteredObject<BinNormalisation>;
END_NAMESPACE_TOMO

#endif
