//
// $Id$
//
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
/*!
  \file
  \ingroup normalisation

  \brief Declaration of class stir::BinNormalisation

  \author Kris Thielemans
  $Date$
  $Revision$
*/

#ifndef __stir_recon_buildblock_BinNormalisation_H__
#define __stir_recon_buildblock_BinNormalisation_H__


#include "stir/RegisteredObject.h"
#include "stir/Bin.h"

START_NAMESPACE_STIR

template <typename elemT> class RelatedViewgrams;
class Succeeded;
class ProjDataInfo;
template <typename T> class shared_ptr;
/*!
  \ingroup normalisation
  \brief Abstract base class for implementing bin-wise normalisation of data.

  As part of the measurement model in PET, there usually is some multiplicative 
  correction for every bin, as in 
  \f[ P^\mathrm{full}_{bv} = \mathrm{norm}_b P^\mathrm{normalised}_{bv} \f]
  This multiplicative correction is usually split in the \c normalisation 
  factors (which are scanner dependent) and the \c attenuation factors (which 
  are object dependent). 

  The present class can be used for both of these factors.
*/
class BinNormalisation : public RegisteredObject<BinNormalisation>
{
public:
  virtual ~BinNormalisation();

  //! initialises the object and checks if it can handle such projection data
  /*! Default version does nothing. */
  virtual Succeeded set_up(const shared_ptr<ProjDataInfo>&);

  //! Return the 'efficiency' factor for a single bin
  /*! With the notation of the class documentation, this returns the factor
    \f$\mathrm{norm}_b \f$. 

    \warning Some derived classes might implement this very inefficiently.
  */
  virtual float get_bin_efficiency(const Bin& bin,const double start_time, const double end_time) const =0;

  //! normalise some data
  /*! 
    This would be used for instance to precorrect unnormalised data. With the
    notation of the class documentation, this would \c divide by the factors 
    \f$\mathrm{norm}_b \f$.

    Default implementation divides with the factors returned by get_bin_efficiency()
    (after applying a threshold to avoid division by 0).
  */
  virtual void apply(RelatedViewgrams<float>&,const double start_time, const double end_time) const;

  //! undo the normalisation of some data
  /*! 
    This would be used for instance to bring geometrically forward projected data to 
    the mean of the measured data. With the
    notation of the class documentation, this would \c multiply by the factors 
    \f$\mathrm{norm}_b \f$.

    Default implementation multiplies with the factors returned by get_bin_efficiency().
  */
  virtual void undo(RelatedViewgrams<float>&,const double start_time, const double end_time) const; 
 
};

END_NAMESPACE_STIR

#endif
