//
// $Id$
//
/*!
  \file
  \ingroup normalisation

  \brief Declaration of class stir::TrivialBinNormalisation

  \author Kris Thielemans
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

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

  virtual void apply(RelatedViewgrams<float>&,const double start_time, const double end_time) const {}
  virtual void undo(RelatedViewgrams<float>&,const double start_time, const double end_time) const {}
  
  virtual float get_bin_efficiency(const Bin& bin,const double start_time, const double end_time) const { return 1;}
  

private:
  virtual void set_defaults() {}
  virtual void initialise_keymap() {}
  
};

END_NAMESPACE_STIR

#endif
