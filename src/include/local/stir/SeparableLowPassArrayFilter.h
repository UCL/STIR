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

#ifndef __Tomo_SeparableLowPassArrayFilter_H__
#define __Tomo_SeparableLowPassArrayFilter_H__

#include "DiscretisedDensity.h"
#include "tomo/SeparableArrayFunctionObject.h"
#include "Array.h"

#include <vector>


#ifndef TOMO_NO_NAMESPACES
using std::vector;
#endif



START_NAMESPACE_TOMO


template <int num_dimensions, typename elemT>
class SeparableLowPassArrayFilter:
      public SeparableArrayFunctionObject <num_dimensions,elemT> 
{
public:  

  //! Default constructor
  SeparableLowPassArrayFilter();  
  
  SeparableLowPassArrayFilter(const VectorWithOffset<elemT>& filter_coefficients);
  
private:
 VectorWithOffset<float> filter_coefficients;
 
};


END_NAMESPACE_TOMO

#endif
