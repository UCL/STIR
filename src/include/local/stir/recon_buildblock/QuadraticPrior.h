//
// $Id$
//
/*!
  \file
  \ingroup recon_buildblock
  \brief Declaration of class QuadraticPriot

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/


#ifndef __stir_recon_buildblock_QuadraticPrior_H__
#define __stir_recon_buildblock_QuadraticPrior_H__


#include "stir/RegisteredParsingObject.h"
#include "stir/recon_buildblock/PriorWithParabolicSurrogate.h"
#include "stir/Array.h"
#include "stir/DiscretisedDensity.h"

START_NAMESPACE_STIR


/*!
  \ingroup recon_buildblock
  \brief
  A class in the GeneralisedPrior hierarchy. This implements a quadratic Gibbs prior.

  \todo make weights used flexible
*/
template <typename elemT>
class QuadraticPrior:  public  
		       RegisteredParsingObject< QuadraticPrior<elemT>,
      		       			        GeneralisedPrior<elemT>,  			      	   
						PriorWithParabolicSurrogate<elemT>
	                                      >
{
public:
  //! Name which will be used when parsing a GeneralisedPrior object
  static const char * const registered_name; 

  //! Default constructor 
  QuadraticPrior();

  //! Constructs it explicitly
  QuadraticPrior(const bool only_2D, float penalization_factor);
  
  //! compute gradient by applying the filter
  void compute_gradient(DiscretisedDensity<3,elemT>& prior_gradient, 
			const DiscretisedDensity<3,elemT> &current_image_estimate);

  //! compute normalised gradient by applying the filter
  // in the case of quadratic priors this will just be the sum of weighting coefficients
  void parabolic_surrogate_curvature(DiscretisedDensity<3,elemT>& parabolic_surrogate_curvature, 
			const DiscretisedDensity<3,elemT> &current_image_estimate);

  //! compute Hessian 
  void compute_Hessian(DiscretisedDensity<3,elemT>& prior_Hessian_for_single_densel, 
		const BasicCoordinate<3,int>& coords,
		const DiscretisedDensity<3,elemT> &current_image_estimate);



  
private:
  bool only_2D;
  Array<3,float> weights;
  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();
  string kappa_filename;
  Array<2,float> precomputed_weights;
  shared_ptr<DiscretisedDensity<3,elemT> > kappa_ptr;
};


END_NAMESPACE_STIR

#endif

