//
// $Id: 
//
/*!

  \file
  \ingroup local/buildblock  
  \brief 
   
    
  \author Sanida Mustafovic
  \author Kris Thielemans
      
   $Date: 
   $Revision: 
*/
/*
    Copyright (C) 2000- 2002, IRSL
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


  BasicCoordinate< num_dimensions,float> fwhm;
  BasicCoordinate< num_dimensions,int> max_kernel_sizes;
 
};


END_NAMESPACE_STIR

#endif
