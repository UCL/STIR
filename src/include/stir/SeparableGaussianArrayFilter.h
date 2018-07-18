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

#include <vector>

START_NAMESPACE_STIR


template <int num_dimensions, typename elemT>
class SeparableGaussianArrayFilter:
      public SeparableArrayFunctionObject <num_dimensions,elemT> 
{
public:  

  //! Default constructor
  SeparableGaussianArrayFilter();  
  
  SeparableGaussianArrayFilter(const BasicCoordinate< num_dimensions,float>&  standard_deviation,
                               const BasicCoordinate< num_dimensions,int>&  number_of_coefficients,
                               bool normalise = false);
  
  
private:

  void construct_filter(bool normalise = false);

  void calculate_coefficients(VectorWithOffset<elemT>& filter_coefficients,
				const int number_of_coefficients,
                const float standard_deviation, bool normalise);


  BasicCoordinate< num_dimensions,float> standard_deviation;
  BasicCoordinate< num_dimensions,int> number_of_coefficients;
 
};


END_NAMESPACE_STIR

#endif
