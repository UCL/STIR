//
// $Id: 
//
/*!

  \file
  \ingroup ImageProcessor
  \brief Declaration of class stir::SeparableGaussianArrayFilter
  \see class stir::SeparableMetzArrayFilter

  \author Kris Thielemans
  \author Ludovica Brusaferri

   $Date: 
   $Revision: 
*/
/*
    Copyright (C) 2000- 2002, IRSL and UCL
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_SeparableGaussianArrayFilter_H__
#define __stir_SeparableGaussianArrayFilter_H__

#include "stir/DiscretisedDensity.h"
#include "stir/SeparableArrayFunctionObject.h"
#include "stir/Array.h"
#include "stir/BasicCoordinate.h"

#include <vector>

START_NAMESPACE_STIR

/*!
  \ingroup Array
  \brief Separable Gaussian filtering in \c n - dimensions

  The implementation follows the same discretisation strategy used for the Metz filter.
  \see SeparableMetzArrayFilter for what a Metz filter is.

  For power 0, the Metz filterthe reduces to the Gaussian filter in frequency-space.
  However, if the FWHM is not a lot larger than the sampling distance it will give small negative values.
  Therefore, if a Gaussian filter is needed, the SeparableGaussianArrayFilter is preferable to a Metz filter
  with power 0.

 */

template <int num_dimensions, typename elemT>
class SeparableGaussianArrayFilter:
      public SeparableArrayFunctionObject <num_dimensions,elemT> 
{
public:  

  //! Default constructor
  SeparableGaussianArrayFilter();  
  

  SeparableGaussianArrayFilter(const BasicCoordinate< num_dimensions,float>&  fwhm,
                               const BasicCoordinate< num_dimensions,int>&  max_kernel_sizes,
                               bool normalise = false);
  
  SeparableGaussianArrayFilter(const float fwhm,
                               const float  max_kernel_sizes,
                               bool normalise = false);
private:

  void construct_filter(bool normalise = false);

  void calculate_coefficients(VectorWithOffset<elemT>& filter_coefficients,
                const int max_kernel_sizes,
                const float fwhm, bool normalise);


  BasicCoordinate< num_dimensions,float> fwhms;
  BasicCoordinate< num_dimensions,int> max_kernel_sizes;
 
};


END_NAMESPACE_STIR

#endif
