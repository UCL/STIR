//
// $Id$
//
/*!
  \file
  \ingroup priors
  \brief Declaration of class PriorWithParabolicSurrogate

  \author Sanida Mustafovic 
  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2002- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/


#ifndef __stir_recon_buildblock_PriorWithParabolicSurrogate_H__
#define __stir_recon_buildblock_PriorWithParabolicSurrogate_H__

#include "stir/recon_buildblock/GeneralisedPrior.h"
#include "stir/BasicCoordinate.h"

START_NAMESPACE_STIR

/*!
  \ingroup priors
  \brief
   this class implements priors with a parabolic surrogate curvature
 
  See for example Erdogan and Fessler, Ordered subsets algorithms for
  transmission tomography, PMB, 44 (1999) 2835.
  
*/
template <typename TargetT>
class PriorWithParabolicSurrogate: 	
  public GeneralisedPrior<TargetT>			      	   
{
public:

  //!this should calculate the parabolic surrogate curvature
  virtual void 
    parabolic_surrogate_curvature(TargetT& parabolic_surrogate_curvature, 
				  const TargetT &current_estimate) = 0;

  //! A function that allows skipping some computations if the curvature is independent of the \c current_estimate
  /*! Defaults to return \c true, but can be overloaded by the derived class.
   */
  virtual bool
    parabolic_surrogate_curvature_depends_on_argument() const
    { return true; }
#if 0
  // TODO this does not work for arbitrary TargetT, but only 3-dimensional things
  // maybe we could have a TargetT::index_type or so
  virtual void 
    compute_Hessian(TargetT& prior_Hessian_for_single_densel, 
		    const BasicCoordinate<3,int>& coords,
		    const TargetT &current_estimate) =0;
#endif

};


END_NAMESPACE_STIR

#endif
