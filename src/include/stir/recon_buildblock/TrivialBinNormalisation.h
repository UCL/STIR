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
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_recon_buildblock_TrivialBinNormalisation_H__
#define __stir_recon_buildblock_TrivialBinNormalisation_H__

#include "stir/recon_buildblock/BinNormalisation.h"
#include "stir/RegisteredParsingObject.h"

START_NAMESPACE_STIR

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

END_NAMESPACE_STIR

#endif
