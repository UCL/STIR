//
// $Id$
//
/*!
  \file
  \ingroup recon_buildblock

  \brief Declaration of class TrivialBinNormalisation

  \author Kris Thielemans
  $Date$
  $Revision$
*/

#ifndef __Tomo_recon_buildblock_TrivialBinNormalisation_H__
#define __Tomo_recon_buildblock_TrivialBinNormalisation_H__

#include "tomo/recon_buildblock/BinNormalisation.h"
#include "tomo/RegisteredParsingObject.h"

START_NAMESPACE_TOMO

/*!
  \ingroup recon_buildblock
  \brief Trivial class which does not do any normalisation whatsoever.
  \todo Make sure that the keyword value \c None corresponds to this class.

*/
class TrivialBinNormalisation : 
   public RegisteredParsingObject<TrivialBinNormalisation, BinNormalisation>
{
public:
  //! Name which will be used when parsing a BinNormalisation object
  static const char * const registered_name; 

  virtual void apply(RelatedViewgrams<float>&) const {}
  virtual void undo(RelatedViewgrams<float>&) const {}

private:
  virtual void set_defaults() {}
  virtual void initialise_keymap() {}
  
};

END_NAMESPACE_TOMO

#endif
