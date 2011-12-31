//
// $Id$
//
/*!

  \file
  \ingroup Array
  \brief Implementations for stir::SeparableArrayFunctionObject

  \author Kris Thielemans
  \author Sanida Mustafovic

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2001- $Date$, Hammersmith Imanet Ltd
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

#include "stir/SeparableArrayFunctionObject.h"
#include "stir/ArrayFunction.h"
#include "stir/is_null_ptr.h"

START_NAMESPACE_STIR

template <int num_dim, typename elemT>
SeparableArrayFunctionObject<num_dim, elemT>::
SeparableArrayFunctionObject()
: all_1d_array_filters(VectorWithOffset< shared_ptr<ArrayFunctionObject<1,elemT> > >(num_dim))
{}

template <int num_dim, typename elemT>
SeparableArrayFunctionObject<num_dim, elemT>::
SeparableArrayFunctionObject(const VectorWithOffset< shared_ptr<ArrayFunctionObject<1,elemT> > >& array_filters)
: all_1d_array_filters(array_filters)
{
  assert(all_1d_array_filters.get_length() == num_dim);
}


template <int num_dim, typename elemT>
void 
SeparableArrayFunctionObject<num_dim, elemT>::
do_it(Array<num_dim,elemT>& array) const
{
  if (!is_trivial())
    {
#ifndef NDEBUG
      // currently in_place_apply_array_functions_on_each_index doesn't handle 0 
      // pointers gracefully, so we check here that there aren't any
      for (typename VectorWithOffset< shared_ptr<ArrayFunctionObject<1,elemT> > >::const_iterator
	      iter=all_1d_array_filters.begin();
	    iter!=all_1d_array_filters.end();
	    ++iter)
	assert(!is_null_ptr(*iter));
#endif
       in_place_apply_array_functions_on_each_index(array, 
						    all_1d_array_filters.begin(), 
						    all_1d_array_filters.end());
    }
}

#if 0
// TODO insert
template <int num_dimensions, typename elemT>
void 
SeparableArrayFunctionObject<num_dimensions, elemT>::
do_it(Array<num_dimensions,elemT>& out_array, const Array<num_dimensions,elemT>& in_array) const
{
   
  //if (!is_trivial())
  {
#ifndef NDEBUG
    // currently apply_array_functions_on_each_index doesn't handle 0 
    // pointers gracefully, so we check here that there aren't any
    for ( VectorWithOffset< shared_ptr<ArrayFunctionObject<1,elemT> > >::const_iterator
	    iter=all_1d_array_filters.begin();
	  iter!=all_1d_array_filters.end();
	    ++iter)
      assert(!is_null_ptr(*iter));
#endif
    apply_array_functions_on_each_index(out_array, in_array,
					all_1d_array_filters.begin(), 
					all_1d_array_filters.end());
  }
  //else somehow copy in_array into out_array but keeping index ranges
   //TODO

}
#endif

template <int num_dim, typename elemT>
bool 
SeparableArrayFunctionObject<num_dim, elemT>::
is_trivial() const
{
  for (typename VectorWithOffset< shared_ptr<ArrayFunctionObject<1,elemT> > >::const_iterator 
	  iter=all_1d_array_filters.begin();
        iter!=all_1d_array_filters.end();
	++iter)
   {
     if (!is_null_ptr(*iter) && !(*iter)->is_trivial())
       return false;
   }
   return true;
}


// instantiation
template class SeparableArrayFunctionObject<3, float>;

END_NAMESPACE_STIR



