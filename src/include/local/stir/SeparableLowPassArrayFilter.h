//
// $Id: 
//
/*!

  \file
  \ingroup buildblock  
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

#ifndef __stir_SeparableLowPassArrayFilter_H__
#define __stir_SeparableLowPassArrayFilter_H__

#include "stir/DiscretisedDensity.h"
#include "stir/SeparableArrayFunctionObject.h"
#include "stir/Array.h"

#include <vector>


#ifndef STIR_NO_NAMESPACES
using std::vector;
#endif



START_NAMESPACE_STIR


template <int num_dimensions, typename elemT>
class SeparableLowPassArrayFilter:
      public SeparableArrayFunctionObject <num_dimensions,elemT> 
{
public:  

  //! Default constructor
  SeparableLowPassArrayFilter();  
  
  SeparableLowPassArrayFilter(const VectorWithOffset<elemT>& filter_coefficients, int z_trivial);
  
private:
 VectorWithOffset<float> filter_coefficients;
 int z_trivial;
};


END_NAMESPACE_STIR

#endif
