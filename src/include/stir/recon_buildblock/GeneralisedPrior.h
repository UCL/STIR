//
// $Id$
//
/*!
  \file
  \ingroup recon_buildblock
  \brief Declaration of class GeneralisedPrior

  \author Kris Thielemans
  \author Sanida Mustafovic

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_recon_buildblock_GeneralisedPrior_H__
#define __stir_recon_buildblock_GeneralisedPrior_H__


#include "stir/RegisteredObject.h"

START_NAMESPACE_STIR


template <int num_dimensions, typename elemT> class DiscretisedDensity;

/*!
  \ingroup recon_buildblock
  \brief
  A base class for 'generalised' priors, i.e. priors for which at least
  a 'gradient' is defined. 

  This class exists to accomodate FilterRootPrior. Otherwise we could
  just live with Prior as a base class.
*/
template <typename elemT>
class GeneralisedPrior: public RegisteredObject<GeneralisedPrior<elemT> >
{
public:
  
  inline GeneralisedPrior(); 
  //! This should compute the gradient of the log of the prior function at the \a current_image_estimate
  /*! The gradient is already multiplied with the penalisation_factor.*/
  virtual void compute_gradient(DiscretisedDensity<3,elemT>& prior_gradient, 
		   const DiscretisedDensity<3,elemT> &current_image_estimate) =0;  
  
  inline float get_penalisation_factor() const;
  inline void set_penalisation_factor(float new_penalisation_factor);

protected:
  float penalisation_factor;

};

END_NAMESPACE_STIR

#include "stir/recon_buildblock/GeneralisedPrior.inl"

#endif
