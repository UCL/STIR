//
// $Id$
//
/*!
  \file
  \ingroup buildblock
  \brief Declaration of class ArrayFunctionObject_2ArgumentImplementation

  \author Kris Thielemans

  \date $Date$
  \version $Revision$
*/

#ifndef __Tomo_ArrayFunctionObject_2ArgumentImplementation_H__
#define __Tomo_ArrayFunctionObject_2ArgumentImplementation_H__


#include "tomo/ArrayFunctionObject.h"


START_NAMESPACE_TOMO

/*!
  \ingroup buildblock
  \brief A convenience class for children of ArrayFunctionObject. It
  implements the in-place operator() in terms of the 2 argument version.

  Sadly, we need to introduce another virtual function for this, as 
  redefining an overloaded function in a derived class, hides all other
  overladed versions. So, we cannot simply leave the 2 arg operator() 
  undefined here. Similarly, we could not only define the 2 arg operator()
  in a derived class.

  \see ArrayFunctionObject_1ArgumentImplementation
*/
template <int num_dimensions, typename elemT>
class ArrayFunctionObject_2ArgumentImplementation :
  public ArrayFunctionObject<num_dimensions,elemT>
{
public:
  //! in-place modification of array, implemented inline
  virtual inline void operator() (Array<num_dimensions,elemT>& array) const
  {
   Array<num_dimensions,elemT> copy_array( array);
   (*this)(array, copy_array);
  }

  virtual void operator() (Array<num_dimensions,elemT>& out_array, 
                           const Array<num_dimensions,elemT>& in_array) const
  {
    do_it(out_array, in_array);
  }
protected:
  virtual void do_it(Array<num_dimensions,elemT>& out_array, 
                     const Array<num_dimensions,elemT>& in_array) const = 0;
};



END_NAMESPACE_TOMO 

#endif 