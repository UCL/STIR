//
// $Id$
//
/*!

  \file
  \ingroup Array
  \brief Declaration of class stir::ArrayFunctionObject_1ArgumentImplementation

  \author Kris Thielemans
  
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for details
*/

#ifndef __stir_ArrayFunctionObject_1ArgumentImplementation_H__
#define __stir_ArrayFunctionObject_1ArgumentImplementation_H__


#include "stir/ArrayFunctionObject.h"
#include "stir/Array.h"

START_NAMESPACE_STIR

/*!
  \ingroup Array
  \brief A convenience class for children of ArrayFunctionObject. It
  implements the 2 argument operator() in terms of the in-place version.

  Sadly, we need to introduce another virtual function for this, as 
  redefining an overloaded function in a derived class, hides all other
  overladed versions. So, we cannot simply leave the 1 arg operator() 
  undefined here. Similarly, we could not only define the 1 arg operator()
  in a derived class.

  \see ArrayFunctionObject_2ArgumentImplementation
*/
template <int num_dimensions, typename elemT>
class ArrayFunctionObject_1ArgumentImplementation :
  public ArrayFunctionObject<num_dimensions,elemT>
{
public:
  virtual void operator() (Array<num_dimensions,elemT>& array) const
  {
    do_it(array);
  }

  //! result stored in another array, implemented inline
  virtual void inline operator() (Array<num_dimensions,elemT>& out_array, 
                           const Array<num_dimensions,elemT>& in_array) const
  {
    assert(out_array.get_index_range() == in_array.get_index_range());
    out_array = in_array;
    do_it(out_array);
  }
protected:
  virtual void do_it(Array<num_dimensions,elemT>& array) const = 0;
};



END_NAMESPACE_STIR 

#endif 
