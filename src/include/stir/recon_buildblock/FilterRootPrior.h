//
// $Id$
//
/*!
  \file
  \ingroup recon_buildblock
  \brief Declaration of class FilterRootPriot

  \author Kris Thielemans
  \author Sanida Mustafovic

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/


#ifndef __stir_recon_buildblock_FilterRootPrior_H__
#define __stir_recon_buildblock_FilterRootPrior_H__


#include "stir/RegisteredParsingObject.h"
#include "stir/recon_buildblock/GeneralisedPrior.h"


START_NAMESPACE_STIR

template <int num_dimensions, typename elemT> class ImageProcessor;
template <int num_dimensions, typename elemT> class DiscretisedDensity;

/*!
  \ingroup recon_buildblock
  \brief
  A class in the GeneralisedPrior hierarchy. This implements 'generalised' 
  priors a la the Median Root Prior (which was invented by
  Sakari Alenius).

  This class takes an ImageProcessor object (i.e. a filter), and computes 
  the prior 'gradient' as
  \f[ G_v = \beta ( {\lambda_v \over F_v} - 1) \f]
  where \f$ \lambda\f$ is the image where to compute the gradient, and
  \f$F\f$ is the image obtained by filtering \f$\lambda\f$.

  Note that for nearly all filters, this is not a real prior, as this
  'gradient' is \e not the gradient of a function. This can be checked
  by computing the 'Hessian' (i.e. the partial derivatives of the components
  of the gradient). For most (interesting) filters, the Hessian will no be
  symmetric.

  The Median Root Prior is obtained by using a MedianImageFilter3D as 
  ImageProcessor.
*/
template <typename elemT>
class FilterRootPrior:  public 
      RegisteredParsingObject<
	      FilterRootPrior<elemT>,
              GeneralisedPrior<elemT>
	       >
{
public:
  //! Name which will be used when parsing a GeneralisedPrior object
  static const char * const registered_name; 

  //! Default constructor (no filter)
  FilterRootPrior();

  //! Constructs it explicitly
  /*! \todo first argument should be shared_ptr */
  FilterRootPrior(ImageProcessor<3,elemT>* filter, float penalization_factor);
  
  //! compute gradient by applying the filter
  void compute_gradient(DiscretisedDensity<3,elemT>& prior_gradient, 
			const DiscretisedDensity<3,elemT> &current_image_estimate);

  
private:
  // TODO should be shared_ptr
   ImageProcessor<3,elemT>* filter_ptr;   
   virtual void set_defaults();
   virtual void initialise_keymap();

};


END_NAMESPACE_STIR

#endif

