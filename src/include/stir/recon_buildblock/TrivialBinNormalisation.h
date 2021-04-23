//
//
/*!
  \file
  \ingroup normalisation

  \brief Declaration of class stir::TrivialBinNormalisation

  \author Kris Thielemans
*/
/*
    Copyright (C) 2000- 2011, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#ifndef __stir_recon_buildblock_TrivialBinNormalisation_H__
#define __stir_recon_buildblock_TrivialBinNormalisation_H__

#include "stir/recon_buildblock/BinNormalisation.h"
#include "stir/RegisteredParsingObject.h"

START_NAMESPACE_STIR

/*!
  \ingroup normalisation
  \brief Trivial class which does not do any normalisation whatsoever.
  \todo Make sure that the keyword value \c None corresponds to this class.

*/
class TrivialBinNormalisation : 
   public RegisteredParsingObject<TrivialBinNormalisation, BinNormalisation>
{
public:
  //! Name which will be used when parsing a BinNormalisation object
  static const char * const registered_name; 

  virtual inline void apply(RelatedViewgrams<float>&,const double start_time, const double end_time) const {}
  virtual inline void undo(RelatedViewgrams<float>&,const double start_time, const double end_time) const {}
  
  virtual inline float get_bin_efficiency(const Bin& bin,const double start_time, const double end_time) const { return 1;}

  virtual inline bool is_trivial() const { return true;}  

private:
  virtual inline void set_defaults() {}
  virtual inline void initialise_keymap() {}
  
};

END_NAMESPACE_STIR

#endif
