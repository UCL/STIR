//
// $Id$
//
/*!
  \file
  \ingroup recon_buildblock
  \brief Declaration of class FilterRootPriot

  \author Kris Thielemans
  \author Sanida Mustafovic

  \date $Date$
  \version $Revision$
*/


#ifndef __Tomo_recon_buildblock_FilterRootPrior_H__
#define __Tomo_recon_buildblock_FilterRootPrior_H__


#include "tomo/RegisteredParsingObject.h"
#include "tomo/recon_buildblock/GeneralisedPrior.h"


START_NAMESPACE_TOMO

template <int num_dimensions, typename elemT> class ImageProcessor;
template <int num_dimensions, typename elemT> class DiscretisedDensity;

/*!
  \ingroup recon_buildblock
  \brief
  A class for generalised priors a la Median Root Prior (invented by
  Sakari Alenius).

  This class takes an ImageProcessor object (i.e. a filter), and computes 
  the prior 'gradient' as
  \f[ G_v = \beta ( {\lambda_v \ over F_v} - 1) \f]
  where \f$ \lambda\f$ is the image where to compute the gradient, and
  \f$F\f$ is the image obtained by filtering \f$\lambda\f$.

  Note that for nearly all filters, this is not a real prior, as this
  'gradient' is \e not the gradient of a function. This can be checked
  by computing the 'Hessian' and seeing if it is symmetric.

  The Median Root Prior is obtained by using a median as ImageProcessor.
*/
template <typename elemT>
class FilterRootPrior:  public 
      RegisteredParsingObject<
	      FilterRootPrior<elemT>,
              GeneralisedPrior<elemT>
	       >
{
public:

  static const char * const registered_name; 

  FilterRootPrior();

  FilterRootPrior(ImageProcessor<3,elemT>* filter, float penalization_factor);
  
  // compute gradient by applying the filter
  void compute_gradient(DiscretisedDensity<3,elemT>& prior_gradient, 
			const DiscretisedDensity<3,elemT> &current_image_estimate);

  
private:
  // TODO should be shared_ptr
   ImageProcessor<3,elemT>* filter_ptr;   
   virtual void set_defaults();
   virtual void initialise_keymap();

};


END_NAMESPACE_TOMO

#endif

