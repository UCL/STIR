//
// $Id$
//
/*!
  \file
  \ingroup recon_buildblock
  \brief Declaration of class PriorWithParabolicSurrogate

  \author Sanida Mustafovic 
  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2002- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/


#ifndef __stir_recon_buildblock_PriorWithParabolicSurrogate_H__
#define __stir_recon_buildblock_PriorWithParabolicSurrogate_H__

#include "stir/recon_buildblock/GeneralisedPrior.h"

START_NAMESPACE_STIR


template <int num_dimensions, typename elemT> class DiscretisedDensity;

/*!
  \ingroup recon_buildblock
  \brief
   this class implements priors with a parabolic surrogate curvature
 
  See for example Erdogan and Fessler, Ordered subsets algorithms for
  transmission tomography, PMB, 44 (1999) 2835.
  
*/
template <typename elemT>
class PriorWithParabolicSurrogate: 	public GeneralisedPrior<elemT>
			      	   
{
public:

  //!this should calculate the parabolic surrogate curvature
  virtual void parabolic_surrogate_curvature(DiscretisedDensity<3,elemT>& parabolic_surrogate_curvature, 
			const DiscretisedDensity<3,elemT> &current_image_estimate) = 0;

};


END_NAMESPACE_STIR

#endif
