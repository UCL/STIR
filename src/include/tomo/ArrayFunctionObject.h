//
// $Id$
//
/*!

  \file
  \ingroup buildblock
  \brief Declaration of class ArrayFunctionObject

  \author Kris Thielemans
  \author Sanida Mustafovic

  \date $Date$
  \version $Revision$
*/

#ifndef __Tomo_ArrayFunctionObject_H__
#define __Tomo_ArrayFunctionObject_H__


#include "Array.h"



START_NAMESPACE_TOMO

/*!
  \ingroup buildblock
  \brief A class for operations on n-dimensional Arrays
*/
template <int num_dimensions, typename elemT>
class ArrayFunctionObject
{
public:
  virtual ~ArrayFunctionObject() {}
  //! in-place modification
  virtual void operator() (Array<num_dimensions,elemT>& array) const = 0;
  //! result stored in another array 
  virtual void operator() (Array<num_dimensions,elemT>& out_array, 
                           const Array<num_dimensions,elemT>& in_array) const = 0;
  //! Should return true when the operations won't modify the object at all
  virtual bool is_trivial() const  = 0;

};




END_NAMESPACE_TOMO 

#endif 
