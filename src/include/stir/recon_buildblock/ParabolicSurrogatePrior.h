//
// $Id$
//
/*!
  \file
  \ingroup recon_buildblock
  \brief Declaration of class ParabolicSurrogatePrior

  \author Sanida Mustafovic 
  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2002- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/


#ifndef __stir_recon_buildblock_ParabolicSurrogatePrior_H__
#define __stir_recon_buildblock_ParabolicSurrogatePrior_H__


#include "stir/RegisteredParsingObject.h"
#include "stir/recon_buildblock/GeneralisedPrior.h"
#include "stir/Array.h"

START_NAMESPACE_STIR


template <int num_dimensions, typename elemT> class DiscretisedDensity;

/*!
  \ingroup recon_buildblock
  \brief
   this class implements parabolic surrogate curvature
 
  
*/
template <typename elemT>
class ParabolicSurrogatePrior: 	public GeneralisedPrior<elemT>
			      	   
{
public:
   //! Default constructor 
  ParabolicSurrogatePrior();

  //!this should calculate the parabolic surrogate curvature
  virtual void parabolic_surrogate_curvature(DiscretisedDensity<3,elemT>& parabolic_surrogate_curvature, 
			const DiscretisedDensity<3,elemT> &current_image_estimate) = 0;
private:  
};


END_NAMESPACE_STIR

#endif