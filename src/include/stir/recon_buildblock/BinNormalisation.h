//
// $Id$
//
/*!
  \file
  \ingroup recon_buildblock

  \brief Declaration of class BinNormalisation

  \author Kris Thielemans
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_recon_buildblock_BinNormalisation_H__
#define __stir_recon_buildblock_BinNormalisation_H__


#include "stir/RegisteredObject.h"

START_NAMESPACE_STIR

template <typename elemT> class RelatedViewgrams;

/*!
  \ingroup recon_buildblock
  \brief Abstract base class for implementing bin-wise normalisation of data.

  As part of the measurement model, there is usually is some multiplicative 
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
  virtual ~BinNormalisation() {}
  //! normalise some data
  /*! 
    This would be used for instance to precorrect unnormalised data. With the
    notation of the class documentation, this would \c divide by the factors 
    \f$\mathrm{norm}_b \f$.
  */
  virtual void apply(RelatedViewgrams<float>&) const= 0;
  //! undo the normalisation of some data
  /*! 
    This would be used for instance to bring geometrically forward projected data to 
    the mean of the measured data. With the
    notation of the class documentation, this would \c multiply by the factors 
    \f$\mathrm{norm}_b \f$.
  */
  virtual void undo(RelatedViewgrams<float>&) const = 0; 
};

END_NAMESPACE_STIR

#endif
